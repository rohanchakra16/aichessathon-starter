#!/usr/bin/env python3
"""Build a byte-reproducible competition ZIP without modifying the official harness."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.package import DEFAULT_INCLUDES, members  # noqa: E402

FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o100644 << 16


def build_deterministic(
    root: Path, destination: Path, includes: tuple[str, ...] = DEFAULT_INCLUDES
) -> list[str]:
    written: list[str] = []
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, name in members(root, includes):
            info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = FILE_MODE
            info.create_system = 3
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
            written.append(name)
    if "agent.py" not in written:
        raise ValueError("submission is missing root-level agent.py")
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("submission.zip"))
    args = parser.parse_args()
    written = build_deterministic(args.root.resolve(), args.output)
    print(f"{args.output} ({args.output.stat().st_size:,} bytes)")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

