#!/usr/bin/env bash
# The style checks of STYLE.md rule 9 (and rule 1, which the one-action
# check enforces), on the paths given or on the whole tree, then rule
# 10's partial order and its no-class-recognition rule over the whole
# package, and rule 11's branch count, which fails on any function over
# five branches.
#
# The interpreter is the active environment's python, which the dev
# extra gives ruff. DECSIM_PYTHON names another interpreter, and
# DECSIM_PYDEPS a folder of packages to put first on its path.
set -u
cd "$(dirname "$0")/.."
python=${DECSIM_PYTHON:-python}
if ! command -v "$python" > /dev/null; then
  echo "no interpreter $python; activate decsim's environment or set" \
    "DECSIM_PYTHON" >&2
  exit 1
fi
if [ -n "${DECSIM_PYDEPS:-}" ]; then
  export PYTHONPATH="$DECSIM_PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
fi
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=(decsim tests tools)
fi
status=0
"$python" -m ruff format --check "${targets[@]}" || status=1
"$python" -m ruff check "${targets[@]}" || status=1
"$python" tools/check_one_action.py "${targets[@]}" || status=1
"$python" tools/check_uses_graph.py decsim || status=1
"$python" tools/check_row_recognition.py decsim || status=1
"$python" -m ruff check --quiet --select C901 \
  --config "lint.mccabe.max-complexity=6" --output-format concise \
  "${targets[@]}" | sed 's/^/over five branches: /'
[ "${PIPESTATUS[0]}" -eq 0 ] || status=1
exit $status
