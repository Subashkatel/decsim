#!/usr/bin/env bash
# The three style checks of REWRITE.md rule 9 (and rule 1, which the
# one-action check enforces), on the paths given or on the whole tree.
#
# The interpreter and the dependency folder default to the checkout's
# own .venv and .pydeps. A git worktree has neither, so point the two
# variables at the checkout that does:
#
#   DECSIM_PYTHON=/path/to/decsim/.venv/bin/python \
#   DECSIM_PYDEPS=/path/to/decsim/.pydeps tools/check.sh
set -u
cd "$(dirname "$0")/.."
python=${DECSIM_PYTHON:-.venv/bin/python}
pydeps=${DECSIM_PYDEPS:-.pydeps}
if [ ! -x "$python" ]; then
  echo "no interpreter at $python; set DECSIM_PYTHON" >&2
  exit 1
fi
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=(decsim tests tools)
fi
status=0
PYTHONPATH=$pydeps "$python" -m ruff format --check "${targets[@]}" || status=1
PYTHONPATH=$pydeps "$python" -m ruff check "${targets[@]}" || status=1
"$python" tools/check_one_action.py "${targets[@]}" || status=1
exit $status
