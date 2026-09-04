"""Phineas 2 bitboard core: flat-array state + move generation + make/unmake.

Hot path = fixed-size numpy int64/uint64 arrays and packed int32 moves only.
No Python objects, no python-chess. Square 0=a1..63=h8, colour 0=white/1=black.

State arrays (owned by weights.p2pos.Position, passed positionally):
  bb    uint64[12]  index = colour*6 + (piece_type-1)
  occ   uint64[3]   [white, black, all]
  mbox  int64[64]   piece index 0..11, or 12 (=empty)
  meta  int64[6]    [side, castling(0..15), ep_sq(0..63 | 64), halfmove, fullmove, phase]

Undo stack (index by ply):
  u_cap  int64[MAXPLY]
  u_meta int64[MAXPLY,6]
  u_zob  uint64[MAXPLY]

Move int32: frm | to<<6 | promo<<12 | kind<<15
  promo 0/2/3/4/5 ; kind 0 quiet 1 dbl-push 2 O-O 3 O-O-O 4 cap 5 ep 6 promo 7 promo-cap
"""

from __future__ import annotations

import os as _os

import numpy as np

from weights import p2tables as _t

MAXPLY = 256
MAXMOVES = 256
CR_WK, CR_WQ, CR_BK, CR_BQ = 1, 2, 4, 8
PIECE_NONE = 12
EP_NONE = 64

KN = _t.KNIGHT_ATT
KG = _t.KING_ATT
PA = _t.PAWN_ATT
RAY = _t.RAY
RAY_POS = _t.RAY_POSITIVE
BISHOP_D = _t.BISHOP_DIRS
ROOK_D = _t.ROOK_DIRS
Z_PSQ = _t.ZOB_PSQ
Z_CASTLE = _t.ZOB_CASTLE
Z_EP = _t.ZOB_EP_FILE
Z_SIDE = _t.ZOB_SIDE
INDEX64 = _t._INDEX64
DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
NOT_FILE_A = _t.NOT_FILE_A
NOT_FILE_H = _t.NOT_FILE_H
IDX_TYPE = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], dtype=np.int64)
PHASE_W = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int64)

ONE = np.uint64(1)

# The De Bruijn bitscan multiply is intentionally modulo 2**64; the pure-Python
# fallback path (numba disabled) would otherwise emit RuntimeWarnings. Numba's
# own uint64 arithmetic wraps silently and is unaffected by this.
np.seterr(over="ignore")

_FORCE_PY = _os.environ.get("P2_NO_NUMBA") == "1"
try:
    if _FORCE_PY:
        raise ImportError
    from numba import njit

    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

    def njit(*a, **k):  # type: ignore[misc]
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=False)
def _lsb(b):
    return INDEX64[np.uint64((b & (~b + ONE)) * DEBRUIJN) >> np.uint64(58)]


@njit(cache=False)
def _msb(b):
    b |= b >> np.uint64(1)
    b |= b >> np.uint64(2)
    b |= b >> np.uint64(4)
    b |= b >> np.uint64(8)
    b |= b >> np.uint64(16)
    b |= b >> np.uint64(32)
    b = b - (b >> np.uint64(1))  # isolate the top set bit
    return INDEX64[np.uint64(b * DEBRUIJN) >> np.uint64(58)]


@njit(cache=False)
def _slider(sq, occ, dirs):
    attacks = np.uint64(0)
    for i in range(dirs.shape[0]):
        d = dirs[i]
        r = RAY[d, sq]
        blockers = r & occ
        if blockers != np.uint64(0):
            first = _lsb(blockers) if RAY_POS[d] == 1 else _msb(blockers)
            r = r ^ RAY[d, first]
        attacks |= r
    return attacks


@njit(cache=False)
def _bishop_att(sq, occ):
    return _slider(sq, occ, BISHOP_D)


@njit(cache=False)
def _rook_att(sq, occ):
    return _slider(sq, occ, ROOK_D)


@njit(cache=False)
def is_attacked(bb, occ_all, sq, by):
    o = by * 6
    if PA[1 - by, sq] & bb[o]:
        return True
    if KN[sq] & bb[o + 1]:
        return True
    if KG[sq] & bb[o + 5]:
        return True
    if _bishop_att(sq, occ_all) & (bb[o + 2] | bb[o + 4]):
        return True
    return (_rook_att(sq, occ_all) & (bb[o + 3] | bb[o + 4])) != np.uint64(0)


@njit(cache=False)
def attackers_to(bb, occ_all, sq, by):
    o = by * 6
    res = PA[1 - by, sq] & bb[o]
    res |= KN[sq] & bb[o + 1]
    res |= KG[sq] & bb[o + 5]
    res |= _bishop_att(sq, occ_all) & (bb[o + 2] | bb[o + 4])
    res |= _rook_att(sq, occ_all) & (bb[o + 3] | bb[o + 4])
    return res


@njit(cache=False)
def king_sq(bb, colour):
    return _lsb(bb[colour * 6 + 5])


@njit(cache=False)
def in_check(bb, occ, meta):
    us = meta[0]
    return is_attacked(bb, occ[2], king_sq(bb, us), 1 - us)


@njit(cache=False)
def _emit(out, n, frm, to, promo, kind):
    out[n] = frm | (to << 6) | (promo << 12) | (kind << 15)
    return n + 1


@njit(cache=False)
def gen_moves(bb, occ, meta, out):
    us = meta[0]
    them = 1 - us
    o = us * 6
    own = occ[us]
    opp = occ[them]
    all_occ = occ[2]
    empty = ~all_occ
    castling = meta[1]
    ep = meta[2]
    n = 0
    pawns = bb[o]

    if us == 0:
        push1 = (pawns << np.uint64(8)) & empty
        push2 = ((push1 & np.uint64(0x0000000000FF0000)) << np.uint64(8)) & empty
        cap7 = (pawns << np.uint64(7)) & NOT_FILE_H & opp
        cap9 = (pawns << np.uint64(9)) & NOT_FILE_A & opp
        promo_rank = np.uint64(0xFF00000000000000)
        fwd = 8
        d7, d9 = -7, -9
    else:
        push1 = (pawns >> np.uint64(8)) & empty
        push2 = ((push1 & np.uint64(0x0000FF0000000000)) >> np.uint64(8)) & empty
        cap7 = (pawns >> np.uint64(7)) & NOT_FILE_A & opp
        cap9 = (pawns >> np.uint64(9)) & NOT_FILE_H & opp
        promo_rank = np.uint64(0x00000000000000FF)
        fwd = -8
        d7, d9 = 7, 9

    b = push1
    while b:
        to = _lsb(b)
        b &= b - ONE
        frm = to - fwd
        if (ONE << np.uint64(to)) & promo_rank:
            n = _emit(out, n, frm, to, 5, 6)
            n = _emit(out, n, frm, to, 4, 6)
            n = _emit(out, n, frm, to, 3, 6)
            n = _emit(out, n, frm, to, 2, 6)
        else:
            n = _emit(out, n, frm, to, 0, 0)
    b = push2
    while b:
        to = _lsb(b)
        b &= b - ONE
        n = _emit(out, n, to - 2 * fwd, to, 0, 1)
    b = cap7
    while b:
        to = _lsb(b)
        b &= b - ONE
        frm = to + d7
        if (ONE << np.uint64(to)) & promo_rank:
            n = _emit(out, n, frm, to, 5, 7)
            n = _emit(out, n, frm, to, 4, 7)
            n = _emit(out, n, frm, to, 3, 7)
            n = _emit(out, n, frm, to, 2, 7)
        else:
            n = _emit(out, n, frm, to, 0, 4)
    b = cap9
    while b:
        to = _lsb(b)
        b &= b - ONE
        frm = to + d9
        if (ONE << np.uint64(to)) & promo_rank:
            n = _emit(out, n, frm, to, 5, 7)
            n = _emit(out, n, frm, to, 4, 7)
            n = _emit(out, n, frm, to, 3, 7)
            n = _emit(out, n, frm, to, 2, 7)
        else:
            n = _emit(out, n, frm, to, 0, 4)
    if ep != EP_NONE:
        att = PA[them, ep] & pawns
        while att:
            frm = _lsb(att)
            att &= att - ONE
            n = _emit(out, n, frm, ep, 0, 5)

    b = bb[o + 1]
    while b:
        frm = _lsb(b)
        b &= b - ONE
        t = KN[frm] & ~own
        while t:
            to = _lsb(t)
            t &= t - ONE
            n = _emit(out, n, frm, to, 0, 4 if (ONE << np.uint64(to)) & opp else 0)

    b = bb[o + 2]
    while b:
        frm = _lsb(b)
        b &= b - ONE
        t = _bishop_att(frm, all_occ) & ~own
        while t:
            to = _lsb(t)
            t &= t - ONE
            n = _emit(out, n, frm, to, 0, 4 if (ONE << np.uint64(to)) & opp else 0)
    b = bb[o + 3]
    while b:
        frm = _lsb(b)
        b &= b - ONE
        t = _rook_att(frm, all_occ) & ~own
        while t:
            to = _lsb(t)
            t &= t - ONE
            n = _emit(out, n, frm, to, 0, 4 if (ONE << np.uint64(to)) & opp else 0)
    b = bb[o + 4]
    while b:
        frm = _lsb(b)
        b &= b - ONE
        t = (_bishop_att(frm, all_occ) | _rook_att(frm, all_occ)) & ~own
        while t:
            to = _lsb(t)
            t &= t - ONE
            n = _emit(out, n, frm, to, 0, 4 if (ONE << np.uint64(to)) & opp else 0)

    frm = king_sq(bb, us)
    t = KG[frm] & ~own
    while t:
        to = _lsb(t)
        t &= t - ONE
        n = _emit(out, n, frm, to, 0, 4 if (ONE << np.uint64(to)) & opp else 0)

    if us == 0:
        k_clr = not is_attacked(bb, all_occ, 4, 1)
        ks_ok = (castling & CR_WK) and not (all_occ & np.uint64(0x60)) and k_clr
        if ks_ok and not is_attacked(bb, all_occ, 5, 1) and not is_attacked(bb, all_occ, 6, 1):
            n = _emit(out, n, 4, 6, 0, 2)
        qs_ok = (castling & CR_WQ) and not (all_occ & np.uint64(0x0E)) and k_clr
        if qs_ok and not is_attacked(bb, all_occ, 3, 1) and not is_attacked(bb, all_occ, 2, 1):
            n = _emit(out, n, 4, 2, 0, 3)
    else:
        k_clr = not is_attacked(bb, all_occ, 60, 0)
        ks_ok = (castling & CR_BK) and not (all_occ & np.uint64(0x6000000000000000)) and k_clr
        if ks_ok and not is_attacked(bb, all_occ, 61, 0) and not is_attacked(bb, all_occ, 62, 0):
            n = _emit(out, n, 60, 62, 0, 2)
        qs_ok = (castling & CR_BQ) and not (all_occ & np.uint64(0x0E00000000000000)) and k_clr
        if qs_ok and not is_attacked(bb, all_occ, 59, 0) and not is_attacked(bb, all_occ, 58, 0):
            n = _emit(out, n, 60, 58, 0, 3)
    return n


CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0] = 15 ^ CR_WQ
CASTLE_MASK[7] = 15 ^ CR_WK
CASTLE_MASK[4] = 15 ^ (CR_WK | CR_WQ)
CASTLE_MASK[56] = 15 ^ CR_BQ
CASTLE_MASK[63] = 15 ^ CR_BK
CASTLE_MASK[60] = 15 ^ (CR_BK | CR_BQ)


@njit(cache=False)
def make(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv):
    frm = mv & 0x3F
    to = (mv >> 6) & 0x3F
    promo = (mv >> 12) & 0x7
    kind = (mv >> 15) & 0xF
    us = meta[0]
    them = 1 - us
    moving = mbox[frm]

    for k in range(6):
        u_meta[ply, k] = meta[k]
    u_zob[ply] = zob[0]

    if meta[2] != EP_NONE:
        zob[0] ^= Z_EP[meta[2] & 7]
    zob[0] ^= Z_CASTLE[meta[1]]

    captured = PIECE_NONE
    cap_sq = to
    if kind == 5:
        cap_sq = to - 8 if us == 0 else to + 8
        captured = mbox[cap_sq]
    elif kind == 4 or kind == 7:
        captured = mbox[to]

    if captured != PIECE_NONE:
        m = ONE << np.uint64(cap_sq)
        bb[captured] ^= m
        occ[them] ^= m
        occ[2] ^= m
        zob[0] ^= Z_PSQ[captured, cap_sq]
        mbox[cap_sq] = PIECE_NONE
        meta[5] -= PHASE_W[IDX_TYPE[captured]]

    mm = ONE << np.uint64(frm)
    mt = ONE << np.uint64(to)
    bb[moving] ^= mm | mt
    occ[us] ^= mm | mt
    occ[2] ^= mm | mt
    zob[0] ^= Z_PSQ[moving, frm] ^ Z_PSQ[moving, to]
    mbox[frm] = PIECE_NONE
    mbox[to] = moving

    if promo != 0:
        newp = us * 6 + (promo - 1)
        bb[moving] ^= mt
        bb[newp] ^= mt
        zob[0] ^= Z_PSQ[moving, to] ^ Z_PSQ[newp, to]
        mbox[to] = newp
        meta[5] += PHASE_W[promo]

    if kind == 2:
        rf, rt, rp = (7, 5, 3) if us == 0 else (63, 61, 9)
        rm = (ONE << np.uint64(rf)) | (ONE << np.uint64(rt))
        bb[rp] ^= rm
        occ[us] ^= rm
        occ[2] ^= rm
        zob[0] ^= Z_PSQ[rp, rf] ^ Z_PSQ[rp, rt]
        mbox[rf] = PIECE_NONE
        mbox[rt] = rp
    elif kind == 3:
        rf, rt, rp = (0, 3, 3) if us == 0 else (56, 59, 9)
        rm = (ONE << np.uint64(rf)) | (ONE << np.uint64(rt))
        bb[rp] ^= rm
        occ[us] ^= rm
        occ[2] ^= rm
        zob[0] ^= Z_PSQ[rp, rf] ^ Z_PSQ[rp, rt]
        mbox[rf] = PIECE_NONE
        mbox[rt] = rp

    new_ep = EP_NONE
    if kind == 1:
        new_ep = frm + 8 if us == 0 else frm - 8
        zob[0] ^= Z_EP[new_ep & 7]
    meta[2] = new_ep

    new_cr = meta[1] & CASTLE_MASK[frm] & CASTLE_MASK[to]
    meta[1] = new_cr
    zob[0] ^= Z_CASTLE[new_cr]

    if captured != PIECE_NONE or IDX_TYPE[moving] == 1:
        meta[3] = 0
    else:
        meta[3] = u_meta[ply, 3] + 1
    if us == 1:
        meta[4] += 1
    meta[0] = them
    zob[0] ^= Z_SIDE
    u_cap[ply] = captured


@njit(cache=False)
def unmake(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv):
    frm = mv & 0x3F
    to = (mv >> 6) & 0x3F
    promo = (mv >> 12) & 0x7
    kind = (mv >> 15) & 0xF
    us = u_meta[ply, 0]
    them = 1 - us

    for k in range(6):
        meta[k] = u_meta[ply, k]
    zob[0] = u_zob[ply]

    moving = mbox[to]
    mt = ONE << np.uint64(to)
    if promo != 0:
        bb[moving] ^= mt
        moving = us * 6
        bb[moving] ^= mt

    mm = ONE << np.uint64(frm)
    bb[moving] ^= mm | mt
    occ[us] ^= mm | mt
    occ[2] ^= mm | mt
    mbox[frm] = moving
    mbox[to] = PIECE_NONE

    captured = u_cap[ply]
    if captured != PIECE_NONE:
        cap_sq = (to - 8 if us == 0 else to + 8) if kind == 5 else to
        m = ONE << np.uint64(cap_sq)
        bb[captured] ^= m
        occ[them] ^= m
        occ[2] ^= m
        mbox[cap_sq] = captured

    if kind == 2:
        rf, rt, rp = (7, 5, 3) if us == 0 else (63, 61, 9)
        rm = (ONE << np.uint64(rf)) | (ONE << np.uint64(rt))
        bb[rp] ^= rm
        occ[us] ^= rm
        occ[2] ^= rm
        mbox[rt] = PIECE_NONE
        mbox[rf] = rp
    elif kind == 3:
        rf, rt, rp = (0, 3, 3) if us == 0 else (56, 59, 9)
        rm = (ONE << np.uint64(rf)) | (ONE << np.uint64(rt))
        bb[rp] ^= rm
        occ[us] ^= rm
        occ[2] ^= rm
        mbox[rt] = PIECE_NONE
        mbox[rf] = rp


@njit(cache=False)
def perft(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, buf, ply, depth):
    if depth == 0:
        return 1
    n = gen_moves(bb, occ, meta, buf[ply])
    us = meta[0]
    total = 0
    for i in range(n):
        mv = buf[ply, i]
        make(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv)
        if not is_attacked(bb, occ[2], king_sq(bb, us), 1 - us):
            total += perft(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, buf, ply + 1, depth - 1)
        unmake(bb, occ, mbox, meta, zob, u_cap, u_meta, u_zob, ply, mv)
    return total
