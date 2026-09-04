"""One-game Phineas 2 worker process, driven over stdin/stdout.

Mirrors the real competition protocol as closely as a dev harness can: a
single process imports agent.py once (paying the numba JIT cost up front,
exactly like the platform), then answers one get_move request per line for
the rest of that process's life -- module state (the TT, our own root-key
history) persists across moves within the game and dies with the process,
same as the real thing.

Protocol (line-based, stdin -> stdout), one game per process:
  in:  "<fen>\t<time_left_ms>\n"
  out: "<uci>\t<score_cp>\t<nodes>\t<depth>\t<elapsed_ms>\n"
  in:  "quit\n"  -> process exits

Errors are reported on stdout as "ERROR\t<repr>" so the driver can record a
crash without losing the pipe.
"""

from __future__ import annotations

import sys
import time


def main() -> None:
    import agent  # import cost happens here, once, like a real game process

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line or line == "quit":
            return
        fen, time_left_str = line.split("\t")
        t0 = time.monotonic()
        try:
            uci = agent.get_move(fen, int(time_left_str))
        except Exception as exc:
            sys.stdout.write(f"ERROR\t{exc!r}\n")
            sys.stdout.flush()
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        info = agent.LAST_INFO
        sys.stdout.write(
            f"{uci}\t{info.get('score', 0)}\t{info.get('nodes', 0)}\t"
            f"{info.get('depth', 0)}\t{elapsed_ms:.1f}\n"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
