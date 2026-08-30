SHELL := /bin/bash

.PHONY: setup play arena zip release-zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package

release-zip:
	uv run python .autoloop/protected/artifact.py --root . --output submission.zip

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
