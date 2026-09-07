#!/usr/bin/env bash
# The three style checks from REWRITE.md rule 7, on the given paths or on
# the whole tree. Run from the repo root.
set -u
cd "$(dirname "$0")/.."
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=(decsim tests tools)
fi
status=0
PYTHONPATH=.pydeps .venv/bin/python -m ruff format --check "${targets[@]}" || status=1
PYTHONPATH=.pydeps .venv/bin/python -m ruff check "${targets[@]}" || status=1
.venv/bin/python tools/check_one_action.py "${targets[@]}" || status=1
exit $status
