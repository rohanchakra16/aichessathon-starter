"""Phineas 2 leaf evaluation.

Faithful to the shipped learned model ``weights/model.json``
(``pairwise_finetuned_tapered_piece_square_evaluator``, schema 5): a tapered
piece-square table blended by remaining-material phase, plus castling-right
terms and the trained bias. Implemented as one njit loop over a signed 12x64
table so no per-square attack generation happens at a leaf.

Deliberate difference from the champion's ``_model_evaluate``: the mobility
term (``MOBILITY_WEIGHT * sum popcount(attacks)``) is dropped. The A/B/C
diagnostic study (exp-0099 workup) showed that term, as implemented, was net
harmful and it costs a second attack pass per leaf. True bitboard mobility is
a later, separately-ablated P4 candidate (last, per the user's stated
priority order, precisely because the old term was net harmful).

P4 candidate 3 adds two small, phase-scaled terms found on top of the PST
already: a per-minor-piece penalty for a piece still sitting on its own home
square (development), and a per-king-adjacent-file bonus for having an own
pawn nearby (a cheap pawn-shield proxy for king safety). Both fade out with
the phase the same way the midgame/endgame PST blend does, since neither
matters much once most pieces are off the board.

P4 candidate 4 adds a passed-pawn term -- the one signal here a flat PST
genuinely cannot express, since whether a pawn is passed depends on where
the *other* pawns are. Unlike development/king-safety it gets *more* weight
toward the endgame, not less: converting a passed pawn is exactly the kind
of "seek progress" signal the user asked for.
"""

from __future__ import annotations

import json
import os

import numpy as np

from weights import p2core as _c

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")

with open(_MODEL_PATH) as _fh:
    _M = json.load(_fh)

_W = np.asarray(_M["weights"], dtype=np.float64)
BIAS = np.int64(round(_M["bias"]))
_MG_OFF, _EG_OFF, _CASTLE_OFF = 0, 384, 768

# Signed 12x64 tables in the p2core piece-index layout (colour*6 + type-1).
# White pieces contribute +weight at the square; black pieces contribute
# -weight at the vertically mirrored square (matches chess.square_mirror).
PST_MG = np.zeros((12, 64), dtype=np.int64)
PST_EG = np.zeros((12, 64), dtype=np.int64)
for _t in range(6):
    for _sq in range(64):
        PST_MG[_t, _sq] = round(_W[_MG_OFF + _t * 64 + _sq])
        PST_EG[_t, _sq] = round(_W[_EG_OFF + _t * 64 + _sq])
        _m = _sq ^ 56
        PST_MG[_t + 6, _sq] = -round(_W[_MG_OFF + _t * 64 + _m])
        PST_EG[_t + 6, _sq] = -round(_W[_EG_OFF + _t * 64 + _m])

CASTLE_K = np.int64(round(_W[_CASTLE_OFF]))
CASTLE_Q = np.int64(round(_W[_CASTLE_OFF + 1]))
MAX_PHASE = np.int64(24)

MATE = np.int64(30000)
MATE_IN_MAX = np.int64(30000 - 512)

# --- king safety / development (P4 candidate 3) --------------------------- #
# Both terms matter with pieces still on the board and fade out with them, so
# both are scaled the same way as the midgame/endgame PST blend (by `phase`),
# not added as a flat bonus. Cheap: piece-index lookups on the already-parsed
# mailbox, no bitboards, no extra full-board passes (king squares are found
# during the existing PST loop below).
DEV_SQ = np.array([1, 2, 5, 6, 57, 58, 61, 62], dtype=np.int64)   # b1 c1 f1 g1 b8 c8 f8 g8
DEV_PIECE = np.array([1, 2, 2, 1, 7, 8, 8, 7], dtype=np.int64)    # WN WB WB WN BN BB BB BN
DEV_SIGN = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int64)
DEV_PENALTY = np.int64(12)   # cp lost per minor piece still on its home square
SHIELD_BONUS = np.int64(10)  # cp gained per king-adjacent file with an own pawn nearby

# --- passed pawns (P4 candidate 4) ----------------------------------------- #
# A flat per-square PST cannot express "this pawn has no enemy pawn ahead of
# it on its own or an adjacent file" -- that depends on where the *other*
# pawns are, not just this one's square -- so it is a genuinely new signal,
# not a duplicate of anything already in the trained table. Value scales up
# steeply with rank (a passed pawn on the 7th is close to a new queen) and,
# unlike development/king-safety, gets *more* weight as material comes off
# rather than less: converting a passed pawn is an endgame concern.
PASSED_BONUS = np.array([0, 0, 6, 12, 22, 40, 70, 0], dtype=np.int64)  # by rank, white orientation

njit = _c.njit


@njit(cache=False, nogil=True)
def _passed_pawn_term(mbox):
    total = np.int64(0)
    for sq in range(64):
        p = mbox[sq]
        if p != 0 and p != 6:
            continue
        f = sq % 8
        r = sq // 8
        blocked = False
        if p == 0:
            for df in (-1, 0, 1):
                ff = f + df
                if ff < 0 or ff > 7:
                    continue
                for rr in range(r + 1, 8):
                    if mbox[rr * 8 + ff] == 6:
                        blocked = True
            if not blocked:
                total += PASSED_BONUS[r]
        else:
            for df in (-1, 0, 1):
                ff = f + df
                if ff < 0 or ff > 7:
                    continue
                for rr in range(0, r):
                    if mbox[rr * 8 + ff] == 0:
                        blocked = True
            if not blocked:
                total -= PASSED_BONUS[7 - r]
    return total


@njit(cache=False, nogil=True)
def _pawn_shield(mbox, king_sq, us):
    kf = king_sq % 8
    kr = king_sq // 8
    fwd = 1 if us == 0 else -1
    pawn_idx = us * 6
    shield = 0
    for df in range(-1, 2):
        f = kf + df
        if f < 0 or f > 7:
            continue
        for dr in (1, 2):
            r = kr + fwd * dr
            if r < 0 or r > 7:
                continue
            if mbox[r * 8 + f] == pawn_idx:
                shield += 1
                break
    return shield


@njit(cache=False, nogil=True)
def evaluate(mbox, meta, pst_mg, pst_eg, castle_k, castle_q, bias, max_phase):
    """Centipawn score from the side-to-move's point of view."""
    mg = np.int64(0)
    eg = np.int64(0)
    wk_sq = 4
    bk_sq = 60
    for sq in range(64):
        p = mbox[sq]
        if p != 12:
            mg += pst_mg[p, sq]
            eg += pst_eg[p, sq]
            if p == 5:
                wk_sq = sq
            elif p == 11:
                bk_sq = sq

    phase = meta[5]
    if phase > max_phase:
        phase = max_phase
    white = (mg * phase + eg * (max_phase - phase)) // max_phase

    cr = meta[1]
    if cr & 1:
        white += castle_k
    if cr & 4:
        white -= castle_k
    if cr & 2:
        white += castle_q
    if cr & 8:
        white -= castle_q

    dev = np.int64(0)
    for i in range(8):
        if mbox[DEV_SQ[i]] == DEV_PIECE[i]:
            dev -= DEV_SIGN[i] * DEV_PENALTY
    shield = (_pawn_shield(mbox, wk_sq, 0) - _pawn_shield(mbox, bk_sq, 1)) * SHIELD_BONUS
    white += (dev + shield) * phase // max_phase

    # passed pawns: half value in the opening, full value by the endgame
    passed = _passed_pawn_term(mbox)
    white += passed * (2 * max_phase - phase) // (2 * max_phase)

    if meta[0] == 0:
        return white + bias
    return -white + bias


def evaluate_py(pos) -> int:
    """Convenience wrapper for tests / the Python reference path."""
    return int(
        evaluate(
            pos.mbox, pos.meta, PST_MG, PST_EG,
            CASTLE_K, CASTLE_Q, BIAS, MAX_PHASE,
        )
    )
