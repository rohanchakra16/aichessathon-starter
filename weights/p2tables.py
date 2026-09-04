"""Precomputed lookup tables for the Phineas 2 bitboard core.

Everything here is built once at import (well inside the 90 s budget) as plain
numpy arrays of unsigned 64-bit words so the numba-jitted search can read them
as frozen module globals. Square 0 is a1, square 63 is h8 (python-chess layout).
Pure Python + numpy; no engine, no network.
"""

from __future__ import annotations

import numpy as np

U64 = np.uint64
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

# piece index in the 12-entry bitboard array: colour * 6 + (piece_type - 1)
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11

FILE_A = np.uint64(0x0101010101010101)
RANK_1 = np.uint64(0x00000000000000FF)
NOT_FILE_A = MASK64 ^ FILE_A
NOT_FILE_H = MASK64 ^ (FILE_A << np.uint64(7))


def _sq(file: int, rank: int) -> int:
    return rank * 8 + file


def _bit(sq: int) -> int:
    return 1 << sq


def _in_board(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


# --- leaper attacks -------------------------------------------------------- #

_KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING_DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

KNIGHT_ATT = np.zeros(64, dtype=U64)
KING_ATT = np.zeros(64, dtype=U64)
PAWN_ATT = np.zeros((2, 64), dtype=U64)  # [colour][square] -> capture targets

for s in range(64):
    f, r = s % 8, s // 8
    kn = 0
    for df, dr in _KNIGHT_DELTAS:
        if _in_board(f + df, r + dr):
            kn |= _bit(_sq(f + df, r + dr))
    KNIGHT_ATT[s] = kn
    kg = 0
    for df, dr in _KING_DELTAS:
        if _in_board(f + df, r + dr):
            kg |= _bit(_sq(f + df, r + dr))
    KING_ATT[s] = kg
    wp = 0
    for df in (-1, 1):
        if _in_board(f + df, r + 1):
            wp |= _bit(_sq(f + df, r + 1))
    PAWN_ATT[WHITE][s] = wp
    bp = 0
    for df in (-1, 1):
        if _in_board(f + df, r - 1):
            bp |= _bit(_sq(f + df, r - 1))
    PAWN_ATT[BLACK][s] = bp


# --- sliding rays --------------------------------------------------------- #
# 8 ray directions. "positive" rays (higher square index) use the LSB of the
# blocker set as the nearest blocker; "negative" rays use the MSB.
#   0:N(+8) 1:NE(+9) 2:E(+1) 3:SE(-7) 4:S(-8) 5:SW(-9) 6:W(-1) 7:NW(+7)
_RAY_DELTAS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
RAY_POSITIVE = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=np.uint8)

RAY = np.zeros((8, 64), dtype=U64)
for d, (df, dr) in enumerate(_RAY_DELTAS):
    for s in range(64):
        f, r = s % 8, s // 8
        bits = 0
        f += df
        r += dr
        while _in_board(f, r):
            bits |= _bit(_sq(f, r))
            f += df
            r += dr
        RAY[d][s] = bits

BISHOP_DIRS = np.array([1, 3, 5, 7], dtype=np.uint8)
ROOK_DIRS = np.array([0, 2, 4, 6], dtype=np.uint8)

# squares strictly between two squares that share a rank/file/diagonal (else 0)
BETWEEN = np.zeros((64, 64), dtype=U64)
for a in range(64):
    for d in range(8):
        ray = int(RAY[d][a])
        walk = ray
        acc = 0
        # step along the ray collecting BETWEEN[a][b] = squares before b
        f, r = a % 8, a // 8
        df, dr = _RAY_DELTAS[d]
        f += df
        r += dr
        acc = 0
        while _in_board(f, r):
            b = _sq(f, r)
            BETWEEN[a][b] = acc
            acc |= _bit(b)
            f += df
            r += dr


# --- Zobrist keys -------------------------------------------------------- #

_rng = np.random.default_rng(0xC0FFEE_2026)
ZOB_PSQ = _rng.integers(0, 1 << 64, size=(12, 64), dtype=U64)
ZOB_CASTLE = _rng.integers(0, 1 << 64, size=16, dtype=U64)
ZOB_EP_FILE = _rng.integers(0, 1 << 64, size=8, dtype=U64)
ZOB_SIDE = U64(_rng.integers(0, 1 << 64, dtype=U64))


# --- bit-scan (De Bruijn) ---------------------------------------------- #

_DEBRUIJN64 = np.uint64(0x03F79D71B4CB0A89)
_INDEX64 = np.zeros(64, dtype=np.int64)
with np.errstate(over="ignore"):
    for i in range(64):
        _INDEX64[int((np.uint64(1 << i) * _DEBRUIJN64) >> np.uint64(58))] = i


# --- tapered piece-square + phase (seeded from the shipped learned model) - #
# Loaded lazily by p2eval; kept here only so the layout constants live together.
PHASE_WEIGHT = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int64)  # by piece_type
MAX_PHASE = 24
