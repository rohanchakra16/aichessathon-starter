"""Hand-verified correctness tests for weights.p2core.see (bitboard SEE)."""

from __future__ import annotations

from weights import p2core as p2c
from weights.p2pos import Position

REAL_CASES: list[tuple[str, str, int, str]] = [
    ("4k3/8/8/8/8/3p4/2P5/4K3 w - - 0 1", "c2d3", 100,
     "free pawn capture, undefended"),
    ("4k3/8/8/8/4p3/3p4/2P5/4K3 w - - 0 1", "c2d3", 0,
     "PxP recaptured by a defending pawn: even trade"),
    ("3qk3/8/8/8/8/3p4/2P5/4K3 w - - 0 1", "c2d3", 0,
     "PxP recaptured by a queen (only attacker): still an even pawn trade"),
    ("4k3/8/8/8/1n6/3p4/2P5/4K3 w - - 0 1", "c2d3", 0,
     "PxP recaptured by a knight (only attacker): even trade"),
    ("4k3/8/8/4p3/3p4/8/8/3QK3 w - - 0 1", "d1d4", 100 - 900,
     "QxP where the pawn is defended by another pawn: queen ends up lost for a pawn"),
    ("4k3/8/8/8/4p3/3p4/2P5/1B2K3 w - - 0 1", "c2d3", 100,
     "3-ply exchange: PxP, PxP, then a bishop on b1 discovered through the "
     "vacated c2 square recaptures -- mops up a second free pawn"),
    ("4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1", "c5d6", 100,
     "en-passant capture of an undefended pawn"),
]


def run() -> bool:
    ok = True
    for fen, uci, expected, desc in REAL_CASES:
        pos = Position.from_fen(fen)
        legal = pos.legal_moves_uci()
        if uci not in legal:
            print(f"  SKIP (illegal in this FEN) {desc}: {uci} not in {legal}")
            continue
        frm = (ord(uci[0]) - 97) + 8 * (int(uci[1]) - 1)
        to = (ord(uci[2]) - 97) + 8 * (int(uci[3]) - 1)
        # find the exact packed move (kind bits) from gen_moves so promo/kind match
        n = int(p2c.gen_moves(pos.bb, pos.occ, pos.meta, pos.buf[0]))
        packed = None
        for i in range(n):
            candidate = int(pos.buf[0, i])
            if (candidate & 0x3F) == frm and ((candidate >> 6) & 0x3F) == to:
                packed = candidate
                break
        assert packed is not None, f"move {uci} not found in gen_moves for {fen}"
        got = int(p2c.see(pos.bb, pos.occ[2], pos.mbox, packed))
        tag = "OK " if got == expected else "BAD"
        if got != expected:
            ok = False
        print(f"  {tag} {desc:45s} see({uci})={got:+5d} want={expected:+5d}")
    return ok


if __name__ == "__main__":
    import sys

    print(f"NUMBA_OK={p2c.NUMBA_OK}")
    result = run()
    print("ALL OK" if result else "FAILURES")
    sys.exit(0 if result else 1)
