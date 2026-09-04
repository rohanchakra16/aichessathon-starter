"""Phineas 2 leaf evaluation.

Faithful to the shipped learned model ``weights/model.json``
(``pairwise_finetuned_tapered_piece_square_evaluator``, schema 5): a tapered
piece-square table blended by remaining-material phase, plus castling-right
terms and the trained bias. Implemented as one njit loop over a signed 12x64
table so no per-square attack generation happens at a leaf.

Deliberate difference from the champion's ``_model_evaluate``: the mobility
term (``MOBILITY_WEIGHT * sum popcount(attacks)``) is dropped. The A/B/C
diagnostic study (exp-0099 workup) showed that term, as implemented, was net
harmful and it costs a second attack pass per leaf. Bitboard mobility can be
reintroduced in P4 if it earns its place in games.
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

njit = _c.njit


@njit(cache=False, nogil=True)
def evaluate(mbox, meta, pst_mg, pst_eg, castle_k, castle_q, bias, max_phase):
    """Centipawn score from the side-to-move's point of view."""
    mg = np.int64(0)
    eg = np.int64(0)
    for sq in range(64):
        p = mbox[sq]
        if p != 12:
            mg += pst_mg[p, sq]
            eg += pst_eg[p, sq]

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
