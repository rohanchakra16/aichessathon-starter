"""Python-side Position: FEN <-> flat arrays, plus a perft driver.

This is the *correctness reference* wrapper. It owns the numpy state arrays and
hands them positionally to the numba-jitted kernels in weights.p2core. Nothing
in here runs inside the search hot path.
"""

from __future__ import annotations

import numpy as np

from weights import p2core as _c
from weights import p2tables as _t

_PIECE_TO_IDX = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}
_IDX_TO_PIECE = {v: k for k, v in _PIECE_TO_IDX.items()}


class Position:
    __slots__ = ("bb", "buf", "mbox", "meta", "occ", "u_cap", "u_meta", "u_zob", "zob")

    def __init__(self) -> None:
        self.bb = np.zeros(12, dtype=np.uint64)
        self.occ = np.zeros(3, dtype=np.uint64)
        self.mbox = np.full(64, _c.PIECE_NONE, dtype=np.int64)
        self.meta = np.zeros(6, dtype=np.int64)
        self.zob = np.zeros(1, dtype=np.uint64)
        self.u_cap = np.zeros(_c.MAXPLY, dtype=np.int64)
        self.u_meta = np.zeros((_c.MAXPLY, 6), dtype=np.int64)
        self.u_zob = np.zeros(_c.MAXPLY, dtype=np.uint64)
        self.buf = np.zeros((_c.MAXPLY, _c.MAXMOVES), dtype=np.int32)

    # --- FEN ---------------------------------------------------------------- #
    @classmethod
    def from_fen(cls, fen: str) -> Position:
        p = cls()
        board, side, castling, ep, half, full = fen.split()
        sq = 56
        for ch in board:
            if ch == "/":
                sq -= 16
            elif ch.isdigit():
                sq += int(ch)
            else:
                idx = _PIECE_TO_IDX[ch]
                p.bb[idx] |= np.uint64(1) << np.uint64(sq)
                p.mbox[sq] = idx
                sq += 1
        p.meta[0] = 0 if side == "w" else 1
        cr = 0
        if "K" in castling:
            cr |= _c.CR_WK
        if "Q" in castling:
            cr |= _c.CR_WQ
        if "k" in castling:
            cr |= _c.CR_BK
        if "q" in castling:
            cr |= _c.CR_BQ
        p.meta[1] = cr
        p.meta[2] = _c.EP_NONE if ep == "-" else (ord(ep[0]) - 97) + 8 * (int(ep[1]) - 1)
        p.meta[3] = int(half)
        p.meta[4] = int(full)
        p._recompute_derived()
        return p

    def to_fen(self) -> str:
        rows = []
        for r in range(7, -1, -1):
            row = ""
            empty = 0
            for f in range(8):
                idx = int(self.mbox[r * 8 + f])
                if idx == _c.PIECE_NONE:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += _IDX_TO_PIECE[idx]
            if empty:
                row += str(empty)
            rows.append(row)
        board = "/".join(rows)
        side = "w" if self.meta[0] == 0 else "b"
        cr = self.meta[1]
        bits = (("K", _c.CR_WK), ("Q", _c.CR_WQ), ("k", _c.CR_BK), ("q", _c.CR_BQ))
        cs = "".join(c for c, b in bits if cr & b) or "-"
        ep = "-"
        if self.meta[2] != _c.EP_NONE:
            e = int(self.meta[2])
            ep = chr(97 + e % 8) + str(e // 8 + 1)
        return f"{board} {side} {cs} {ep} {self.meta[3]} {self.meta[4]}"

    def _recompute_derived(self) -> None:
        w = np.uint64(0)
        b = np.uint64(0)
        for i in range(6):
            w |= self.bb[i]
        for i in range(6, 12):
            b |= self.bb[i]
        self.occ[0] = w
        self.occ[1] = b
        self.occ[2] = w | b
        # phase
        phase = 0
        for i in range(12):
            bbi = int(self.bb[i])
            cnt = bin(bbi).count("1")
            phase += cnt * int(_t.PHASE_WEIGHT[_c.IDX_TYPE[i]])
        self.meta[5] = phase
        self.zob[0] = self._zobrist_from_scratch()

    def _zobrist_from_scratch(self) -> np.uint64:
        z = np.uint64(0)
        for idx in range(12):
            bbi = int(self.bb[idx])
            while bbi:
                s = (bbi & -bbi).bit_length() - 1
                z ^= _t.ZOB_PSQ[idx, s]
                bbi &= bbi - 1
        z ^= _t.ZOB_CASTLE[int(self.meta[1])]
        if self.meta[2] != _c.EP_NONE:
            z ^= _t.ZOB_EP_FILE[int(self.meta[2]) & 7]
        if self.meta[0] == 1:
            z ^= _t.ZOB_SIDE
        return z

    # --- perft ------------------------------------------------------------- #
    def perft(self, depth: int) -> int:
        return int(
            _c.perft(
                self.bb, self.occ, self.mbox, self.meta, self.zob,
                self.u_cap, self.u_meta, self.u_zob, self.buf, 0, depth,
            )
        )

    def legal_moves_uci(self) -> list[str]:
        n = int(_c.gen_moves(self.bb, self.occ, self.meta, self.buf[0]))
        us = int(self.meta[0])
        out = []
        for i in range(n):
            mv = int(self.buf[0, i])
            _c.make(self.bb, self.occ, self.mbox, self.meta, self.zob,
                   self.u_cap, self.u_meta, self.u_zob, 0, mv)
            legal = not _c.is_attacked(self.bb, self.occ[2], _c.king_sq(self.bb, us), 1 - us)
            _c.unmake(self.bb, self.occ, self.mbox, self.meta, self.zob,
                     self.u_cap, self.u_meta, self.u_zob, 0, mv)
            if legal:
                out.append(move_to_uci(mv))
        return sorted(out)


def move_to_uci(mv: int) -> str:
    frm = mv & 0x3F
    to = (mv >> 6) & 0x3F
    promo = (mv >> 12) & 0x7
    s = _sq_name(frm) + _sq_name(to)
    if promo:
        s += {2: "n", 3: "b", 4: "r", 5: "q"}[promo]
    return s


def _sq_name(s: int) -> str:
    return chr(97 + s % 8) + str(s // 8 + 1)
