"""Phineas 2 search: njit iterative-deepening negamax / PVS with a hash TT.

Alpha-beta + principal-variation search, transposition table (hash key,
depth stored in the entry, bucketed, generation-aged, never fully cleared),
move ordering (TT move / SEE-ordered captures / two killers / history), a
quiescence search (stand-pat + SEE-non-losing captures + delta pruning), and
pruning (mate-distance, reverse-futility, null-move, frontier futility, LMR).

The Python driver owns iterative deepening and the wall clock; a watchdog
thread flips ``stop[0]`` and the njit tree reads it every 2048 nodes.
"""

from __future__ import annotations

import threading

import numpy as np

from weights import p2core as _c
from weights import p2eval as _e

njit = _c.njit

MATE = 30000
MATE_IN_MAX = MATE - 512
INF = 32000

TT_BITS = 21
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)

# flag values in a TT entry
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# how many plies of our own game history to carry in for repetition detection
PREFIX_CAP = 96
PLY_LIMIT = _c.MAXPLY - 2

# Zobrist components needed for the null move (read as module globals by njit).
_ZS = _c.Z_SIDE
_ZEP = _c.Z_EP
_EPN = _c.EP_NONE

RFP_MARGIN = 85       # reverse-futility: eval - RFP_MARGIN*depth >= beta -> prune
FUT_MARGIN = 120      # frontier futility for quiet moves
NMP_MIN_PHASE = 3     # skip null move in near-zugzwang (very few pieces)


class Searcher:
    """Owns the persistent search state (TT survives between moves in a game)."""

    def __init__(self) -> None:
        self.tt_key = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_move = np.zeros(TT_SIZE, dtype=np.int32)
        self.tt_score = np.zeros(TT_SIZE, dtype=np.int32)
        self.tt_depth = np.full(TT_SIZE, -1, dtype=np.int16)
        self.tt_flag = np.zeros(TT_SIZE, dtype=np.int8)
        self.tt_gen = np.zeros(TT_SIZE, dtype=np.int16)
        self.generation = 0
        self.buf = np.zeros((_c.MAXPLY, _c.MAXMOVES), dtype=np.int32)
        self.mscore = np.zeros((_c.MAXPLY, _c.MAXMOVES), dtype=np.int64)
        self.killers = np.zeros((_c.MAXPLY, 2), dtype=np.int32)
        self.history = np.zeros((12, 64), dtype=np.int64)
        self.hist_keys = np.zeros(PREFIX_CAP + _c.MAXPLY + 8, dtype=np.uint64)
        self.nodes = np.zeros(1, dtype=np.int64)
        self.stop = np.zeros(1, dtype=np.int64)
        self.pv = np.zeros(_c.MAXPLY, dtype=np.int32)

    def new_game(self) -> None:
        self.tt_key.fill(0)
        self.tt_depth.fill(-1)
        self.generation = 0
        self.history.fill(0)

    # -- driver ---------------------------------------------------------- #
    def search(self, pos, max_depth: int = 64, time_ms: float = 1000.0,
               hist_prefix: np.ndarray | None = None) -> tuple[str, int, int, int]:
        """Returns (best_move_uci, score_cp, nodes, completed_depth)."""
        from weights.p2pos import move_to_uci

        self.generation = (self.generation + 1) & 0x7FFF
        self.killers.fill(0)
        self.history //= 2
        self.nodes[0] = 0
        self.stop[0] = 0

        nprefix = 0
        if hist_prefix is not None and len(hist_prefix) > 0:
            tail = hist_prefix[-PREFIX_CAP:]
            nprefix = len(tail)
            self.hist_keys[:nprefix] = tail
        self.hist_keys[nprefix] = pos.zob[0]

        timer = threading.Timer(time_ms / 1000.0, lambda: self.stop.__setitem__(0, 1))
        timer.daemon = True
        timer.start()

        best_move = 0
        best_score = 0
        completed_depth = 0
        try:
            for depth in range(1, max_depth + 1):
                score = _search_root(
                    pos.bb, pos.occ, pos.mbox, pos.meta, pos.zob,
                    pos.u_cap, pos.u_meta, pos.u_zob,
                    self.buf, self.mscore, self.killers, self.history,
                    self.hist_keys, nprefix,
                    self.tt_key, self.tt_move, self.tt_score, self.tt_depth,
                    self.tt_flag, self.tt_gen, self.generation,
                    self.nodes, self.stop, self.pv, depth,
                    _e.PST_MG, _e.PST_EG, _e.CASTLE_K, _e.CASTLE_Q, _e.BIAS, _e.MAX_PHASE,
                )
                if self.stop[0]:
                    # depth aborted mid-way; keep the previous completed result
                    if self.pv[0] != 0 and depth == 1:
                        best_move = int(self.pv[0])
                        best_score = int(score)
                        completed_depth = 1
                    break
                best_move = int(self.pv[0])
                best_score = int(score)
                completed_depth = depth
                if abs(best_score) >= MATE_IN_MAX:
                    break
        finally:
            timer.cancel()

        if best_move == 0:
            n = int(_c.gen_moves(pos.bb, pos.occ, pos.meta, self.buf[0]))
            us = int(pos.meta[0])
            for i in range(n):
                mv = int(self.buf[0, i])
                _c.make(pos.bb, pos.occ, pos.mbox, pos.meta, pos.zob,
                        pos.u_cap, pos.u_meta, pos.u_zob, 0, mv)
                legal = not _c.is_attacked(pos.bb, pos.occ[2], _c.king_sq(pos.bb, us), 1 - us)
                _c.unmake(pos.bb, pos.occ, pos.mbox, pos.meta, pos.zob,
                          pos.u_cap, pos.u_meta, pos.u_zob, 0, mv)
                if legal:
                    best_move = mv
                    break

        return move_to_uci(best_move), best_score, int(self.nodes[0]), completed_depth


@njit(cache=False, nogil=True)
def _score_moves(bb, occ, mbox, buf, mscore, ply, n, tt_move, killers, history):
    """Order captures by actual SEE (swap-off) value, not MVV-LVA: a capture
    that loses material once every recapture is played out scores *below*
    ordinary quiet moves instead of ahead of them, so the search stops
    wasting early nodes -- and reduction/pruning budget -- trying piece
    hangs before it tries plausible quiet moves."""
    occ_all = occ[2]
    for i in range(n):
        mv = buf[ply, i]
        kind = (mv >> 15) & 0xF
        if mv == tt_move:
            mscore[ply, i] = 1_000_000_000
        elif kind == 4 or kind == 5 or kind == 7:
            s = _c.see(bb, occ_all, mbox, mv)
            mscore[ply, i] = 500_000_000 + s if s >= 0 else s
        elif kind == 6:  # quiet promotion
            mscore[ply, i] = 400_000_000 + ((mv >> 12) & 0x7)
        elif mv == killers[ply, 0]:
            mscore[ply, i] = 300_000_002
        elif mv == killers[ply, 1]:
            mscore[ply, i] = 300_000_001
        else:
            frm = mv & 0x3F
            to = (mv >> 6) & 0x3F
            p = mbox[frm]
            mscore[ply, i] = history[p, to] if p != 12 else 0


@njit(cache=False, nogil=True)
def _pick_move(buf, mscore, ply, n, start):
    best = start
    for i in range(start + 1, n):
        if mscore[ply, i] > mscore[ply, best]:
            best = i
    if best != start:
        tm = buf[ply, start]
        buf[ply, start] = buf[ply, best]
        buf[ply, best] = tm
        ts = mscore[ply, start]
        mscore[ply, start] = mscore[ply, best]
        mscore[ply, best] = ts
    return buf[ply, start]


@njit(cache=False, nogil=True)
def _is_repetition(hist_keys, base, ply, zob, halfmove):
    # scan backwards over the reversible window, both search and game history
    count = 0
    i = base + ply - 2
    stop_at = base + ply - halfmove
    if stop_at < 0:
        stop_at = 0
    while i >= stop_at:
        if hist_keys[i] == zob:
            count += 1
            if count >= 1:
                return True
        i -= 2
    return False


@njit(cache=False, nogil=True)
def _quiesce(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
             buf, mscore, killers, history, ply, alpha, beta,
             nodes, stop, pst_mg, pst_eg, ck, cq, bias, maxph):
    nodes[0] += 1
    if nodes[0] & 2047 == 0 and stop[0] != 0:
        return 0

    stand = _e.evaluate(mbox, meta, bb, occ, pst_mg, pst_eg, ck, cq, bias, maxph)
    if ply >= PLY_LIMIT:
        return stand
    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand

    us = meta[0]
    occ_all = occ[2]
    n = _c.gen_moves(bb, occ, meta, buf[ply])
    # order: captures only, by actual SEE value (not MVV-LVA) so a losing
    # capture sorts last and gets skipped outright instead of merely
    # de-prioritised -- this is what stops the search wasting quiescence
    # nodes (and horizon) on hanging a piece into a well-defended square.
    for i in range(n):
        mv = buf[ply, i]
        kind = (mv >> 15) & 0xF
        if kind == 4 or kind == 5 or kind == 7:
            mscore[ply, i] = _c.see(bb, occ_all, mbox, mv)
        else:
            mscore[ply, i] = -1_000_000

    for idx in range(n):
        mv = _pick_move(buf, mscore, ply, n, idx)
        see_val = mscore[ply, idx]
        if see_val < 0:
            break  # remaining entries are losing captures or non-captures
        kind = (mv >> 15) & 0xF
        # delta pruning: even this capture's real (SEE) gain can't reach alpha
        if stand + see_val + 200 < alpha and kind != 7:
            continue
        _c.make(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv)
        if _c.is_attacked(bb, occ[2], _c.king_sq(bb, us), 1 - us):
            _c.unmake(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv)
            continue
        score = -_quiesce(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                          buf, mscore, killers, history, ply + 1, -beta, -alpha,
                          nodes, stop, pst_mg, pst_eg, ck, cq, bias, maxph)
        _c.unmake(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False, nogil=True)
def _negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
             buf, mscore, killers, history, hist_keys, base,
             tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
             nodes, stop, ply, depth, alpha, beta,
             pst_mg, pst_eg, ck, cq, bias, maxph):
    nodes[0] += 1
    if nodes[0] & 2047 == 0 and stop[0] != 0:
        return 0

    us = meta[0]
    if ply >= PLY_LIMIT:
        return _e.evaluate(mbox, meta, bb, occ, pst_mg, pst_eg, ck, cq, bias, maxph)

    hist_keys[base + ply] = zob[0]
    if ply > 0 and meta[3] >= 4 and _is_repetition(hist_keys, base, ply, zob[0], meta[3]):
        return 0
    if meta[3] >= 100:
        return 0

    # mate-distance pruning
    if ply > 0:
        if -MATE + ply > alpha:
            alpha = -MATE + ply
        if MATE - ply < beta:
            beta = MATE - ply
        if alpha >= beta:
            return alpha

    in_chk = _c.is_attacked(bb, occ[2], _c.king_sq(bb, us), 1 - us)
    if in_chk:
        depth += 1

    if depth <= 0:
        return _quiesce(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                        buf, mscore, killers, history, ply, alpha, beta,
                        nodes, stop, pst_mg, pst_eg, ck, cq, bias, maxph)

    # -- TT probe --
    key = zob[0]
    slot = np.int64(key & TT_MASK)
    tt_mv = np.int32(0)
    if tt_key[slot] == key and tt_depth[slot] >= 0:
        tt_mv = tt_move[slot]
        if ply > 0 and tt_depth[slot] >= depth:
            s = tt_score[slot]
            if s > MATE_IN_MAX:
                s -= ply
            elif s < -MATE_IN_MAX:
                s += ply
            f = tt_flag[slot]
            if f == TT_EXACT:
                return s
            if f == TT_LOWER and s > alpha:
                alpha = s
            elif f == TT_UPPER and s < beta:
                beta = s
            if alpha >= beta:
                return s

    non_pv = beta - alpha == 1
    eval_static = _e.evaluate(mbox, meta, bb, occ, pst_mg, pst_eg, ck, cq, bias, maxph)

    # reverse futility / static null-move pruning
    if (non_pv and not in_chk and depth <= 6 and beta > -MATE_IN_MAX
            and eval_static - RFP_MARGIN * depth >= beta):
        return eval_static

    # null-move pruning
    if (non_pv and not in_chk and depth >= 3 and eval_static >= beta
            and meta[5] > NMP_MIN_PHASE and beta > -MATE_IN_MAX):
        s0 = meta[0]
        s2 = meta[2]
        s3 = meta[3]
        sz = zob[0]
        meta[0] = 1 - s0
        if s2 != _EPN:
            zob[0] ^= _ZEP[s2 & 7]
            meta[2] = _EPN
        zob[0] ^= _ZS
        meta[3] = s3 + 1
        r = 3 + depth // 6
        nd = depth - 1 - r
        if nd < 0:
            nd = 0
        null_score = -_negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                               buf, mscore, killers, history, hist_keys, base,
                               tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                               nodes, stop, ply + 1, nd, -beta, -beta + 1,
                               pst_mg, pst_eg, ck, cq, bias, maxph)
        meta[0] = s0
        meta[2] = s2
        meta[3] = s3
        zob[0] = sz
        if stop[0] != 0:
            return 0
        if null_score >= beta:
            return beta

    n = _c.gen_moves(bb, occ, meta, buf[ply])
    _score_moves(bb, occ, mbox, buf, mscore, ply, n, tt_mv, killers, history)

    old_alpha = alpha
    best_score = -INF
    best_move = np.int32(0)
    legal = 0
    quiets_tried = 0
    for idx in range(n):
        mv = _pick_move(buf, mscore, ply, n, idx)
        kind = (mv >> 15) & 0xF
        is_quiet = kind == 0 or kind == 1 or kind == 6

        # frontier futility pruning of quiet moves
        if (is_quiet and legal >= 1 and depth <= 2 and not in_chk
                and best_score > -MATE_IN_MAX
                and eval_static + FUT_MARGIN * depth <= alpha):
            quiets_tried += 1
            continue

        _c.make(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv)
        if _c.is_attacked(bb, occ[2], _c.king_sq(bb, us), 1 - us):
            _c.unmake(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv)
            continue
        legal += 1
        if is_quiet:
            quiets_tried += 1

        new_depth = depth - 1
        # late move reduction
        reduction = 0
        if (depth >= 3 and legal >= 4 and is_quiet and not in_chk
                and mv != killers[ply, 0] and mv != killers[ply, 1]):
            reduction = 1
            if quiets_tried >= 8:
                reduction += 1
            if depth >= 9:
                reduction += 1

        if legal == 1:
            score = -_negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                              buf, mscore, killers, history, hist_keys, base,
                              tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                              nodes, stop, ply + 1, new_depth, -beta, -alpha,
                              pst_mg, pst_eg, ck, cq, bias, maxph)
        else:
            rd = new_depth - reduction
            if rd < 0:
                rd = 0
            score = -_negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                              buf, mscore, killers, history, hist_keys, base,
                              tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                              nodes, stop, ply + 1, rd, -alpha - 1, -alpha,
                              pst_mg, pst_eg, ck, cq, bias, maxph)
            if score > alpha and reduction > 0:
                score = -_negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                                  buf, mscore, killers, history, hist_keys, base,
                                  tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                                  nodes, stop, ply + 1, new_depth, -alpha - 1, -alpha,
                                  pst_mg, pst_eg, ck, cq, bias, maxph)
            if alpha < score < beta:
                score = -_negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                                  buf, mscore, killers, history, hist_keys, base,
                                  tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                                  nodes, stop, ply + 1, new_depth, -beta, -alpha,
                                  pst_mg, pst_eg, ck, cq, bias, maxph)
        _c.unmake(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv)

        if stop[0] != 0:
            return 0

        if score > best_score:
            best_score = score
            best_move = mv
        if score > alpha:
            alpha = score
        if alpha >= beta:
            kind = (mv >> 15) & 0xF
            if kind != 4 and kind != 5 and kind != 7:
                if killers[ply, 0] != mv:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = mv
                frm = mv & 0x3F
                to = (mv >> 6) & 0x3F
                p = mbox[frm]
                if p != 12:
                    history[p, to] += depth * depth
            break

    if legal == 0:
        if in_chk:
            return -MATE + ply
        return 0

    # -- TT store --
    store = best_score
    if store > MATE_IN_MAX:
        store += ply
    elif store < -MATE_IN_MAX:
        store -= ply
    if tt_key[slot] != key or depth >= tt_depth[slot] or tt_gen[slot] != gen:
        tt_key[slot] = key
        tt_move[slot] = best_move
        tt_score[slot] = store
        tt_depth[slot] = depth
        tt_gen[slot] = gen
        if best_score <= old_alpha:
            tt_flag[slot] = TT_UPPER
        elif best_score >= beta:
            tt_flag[slot] = TT_LOWER
        else:
            tt_flag[slot] = TT_EXACT

    return best_score


@njit(cache=False, nogil=True)
def _search_root(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                 buf, mscore, killers, history, hist_keys, base,
                 tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                 nodes, stop, pv, depth,
                 pst_mg, pst_eg, ck, cq, bias, maxph):
    alpha = -INF
    beta = INF
    us = meta[0]
    n = _c.gen_moves(bb, occ, meta, buf[0])

    key = zob[0]
    slot = np.int64(key & TT_MASK)
    tt_mv = tt_move[slot] if (tt_key[slot] == key and tt_depth[slot] >= 0) else np.int32(0)
    _score_moves(bb, occ, mbox, buf, mscore, 0, n, tt_mv, killers, history)

    best_score = -INF
    best_move = np.int32(0)
    legal = 0
    for idx in range(n):
        mv = _pick_move(buf, mscore, 0, n, idx)
        _c.make(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, 0, mv)
        if _c.is_attacked(bb, occ[2], _c.king_sq(bb, us), 1 - us):
            _c.unmake(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, 0, mv)
            continue
        legal += 1
        if legal == 1:
            score = -_negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                              buf, mscore, killers, history, hist_keys, base,
                              tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                              nodes, stop, 1, depth - 1, -beta, -alpha,
                              pst_mg, pst_eg, ck, cq, bias, maxph)
        else:
            score = -_negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                              buf, mscore, killers, history, hist_keys, base,
                              tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                              nodes, stop, 1, depth - 1, -alpha - 1, -alpha,
                              pst_mg, pst_eg, ck, cq, bias, maxph)
            if alpha < score < beta:
                score = -_negamax(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob,
                                  buf, mscore, killers, history, hist_keys, base,
                                  tt_key, tt_move, tt_score, tt_depth, tt_flag, tt_gen, gen,
                                  nodes, stop, 1, depth - 1, -beta, -alpha,
                                  pst_mg, pst_eg, ck, cq, bias, maxph)
        _c.unmake(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, 0, mv)

        if stop[0] != 0:
            break
        if score > best_score:
            best_score = score
            best_move = mv
            pv[0] = mv
        if score > alpha:
            alpha = score

    if best_move != 0:
        slot2 = np.int64(key & TT_MASK)
        store = best_score
        if store > MATE_IN_MAX:
            store += 0
        tt_key[slot2] = key
        tt_move[slot2] = best_move
        tt_score[slot2] = store
        tt_depth[slot2] = depth
        tt_flag[slot2] = TT_EXACT
        tt_gen[slot2] = gen
        pv[0] = best_move
    return best_score
