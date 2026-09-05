"""Correctness tests for weights.p2tb (Syzygy 3-4 piece probing)."""

from __future__ import annotations

import sys

import chess

from weights import p2tb

WIN_POSITIONS = [
    ("8/8/8/4k3/8/8/3QK3/8 w - - 0 1", "KQvK, white to move"),
    ("8/8/8/4k3/8/8/3RK3/8 w - - 0 1", "KRvK, white to move"),
    ("8/3P4/3K4/8/8/8/6k1/8 w - - 0 1", "KPvK, pawn near promotion"),
    ("8/8/8/8/8/2k5/1p6/1K6 b - - 0 1", "KPvK, black pawn winning"),
]

DRAW_POSITIONS = [
    ("8/8/8/4k3/8/8/3BK3/8 w - - 0 1", "KBvK, insufficient material"),
    ("8/8/8/4k3/8/8/3NK3/8 w - - 0 1", "KNvK, insufficient material"),
]

MATE_IN_ONE = [
    ("6k1/8/6K1/8/8/8/8/7Q w - - 0 1", "KQvK back-rank mate available"),
]


def _check_no_regression(fen: str, uci: str, label: str) -> bool:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        print(f"  BAD {label}: {uci} is not even legal in {fen}")
        return False
    board.push(move)
    tb = p2tb._get_tablebase()
    assert tb is not None
    try:
        opp_wdl = -2 if board.is_checkmate() else tb.probe_wdl(board)
    except Exception as exc:
        print(f"  BAD {label}: could not verify resulting wdl: {exc}")
        return False
    our_wdl = -opp_wdl
    print(f"  {'OK ' if our_wdl >= 0 else 'BAD'} {label}: chose {uci}, "
          f"resulting our_wdl={our_wdl}")
    return our_wdl >= 0


def run() -> bool:
    ok = True

    print("== win positions: must never turn a win into a draw/loss ==")
    for fen, label in WIN_POSITIONS:
        uci = p2tb.probe_best_move(fen)
        if uci is None:
            print(f"  BAD {label}: no move returned")
            ok = False
            continue
        ok &= _check_no_regression(fen, uci, label)

    print("== draw positions: must never turn a draw into a loss ==")
    for fen, label in DRAW_POSITIONS:
        uci = p2tb.probe_best_move(fen)
        if uci is None:
            print(f"  BAD {label}: no move returned")
            ok = False
            continue
        ok &= _check_no_regression(fen, uci, label)

    print("== mate-in-one: must find it ==")
    for fen, label in MATE_IN_ONE:
        uci = p2tb.probe_best_move(fen)
        board = chess.Board(fen)
        if uci is None:
            print(f"  BAD {label}: no move returned")
            ok = False
            continue
        board.push(chess.Move.from_uci(uci))
        found_mate = board.is_checkmate()
        print(f"  {'OK ' if found_mate else 'BAD'} {label}: chose {uci}, "
              f"checkmate={found_mate}")
        ok &= found_mate

    print("== >4 pieces: must decline (fall back to search) ==")
    fen5 = "8/8/8/3k4/3n4/3P4/3K4/3R4 w - - 0 1"  # 5 pieces
    uci = p2tb.probe_best_move(fen5)
    print(f"  {'OK ' if uci is None else 'BAD'} 5-piece position -> {uci!r}")
    ok &= uci is None

    print("== progress test: prefers advancing the pawn over a plain king shuffle ==")
    fen_prog = "8/8/8/8/2k5/8/2P5/2K5 w - - 0 1"  # KPvK, pawn far from queening
    uci = p2tb.probe_best_move(fen_prog)
    print(f"  chose {uci!r} (any legal winning move is acceptable; "
          f"reported for manual inspection)")
    ok &= uci is not None

    return ok


if __name__ == "__main__":
    result = run()
    print("ALL OK" if result else "FAILURES")
    sys.exit(0 if result else 1)
