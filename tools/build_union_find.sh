#!/usr/bin/env bash
# The compiled Union-Find row, built into the package folder beside its
# sources: the decoder and the cluster gap its confidence signal walks,
# in one library. The plain build is what decsim loads; the sanitized
# build is a second library the corpus runs under to prove the C stays
# inside its buffers, and it is never the one a run uses.
#
#   tools/build_union_find.sh              -> union_find.so
#   tools/build_union_find.sh --sanitize   -> union_find_sanitized.so
#
# The undefined-behaviour sanitizer traps rather than reports, because
# its runtime library ships separately from the compiler and a node
# that has gcc need not have it; a trap still stops the process on the
# first undefined operation.
#
# The compiler defaults to gcc and CC overrides it.
set -eu
cd "$(dirname "$0")/.."
compiler=${CC:-gcc}
sources=(decsim/decoders/union_find/union_find.c)
sources+=(decsim/decoders/union_find/cluster_gap.c)
flags=(-std=c11 -Wall -Wextra -Wpedantic -Werror -fPIC -shared)
library=decsim/decoders/union_find/union_find.so
if [ "${1:-}" = "--sanitize" ]; then
  library=decsim/decoders/union_find/union_find_sanitized.so
  flags+=(-O1 -g -fsanitize=address,undefined)
  flags+=(-fsanitize-undefined-trap-on-error -fno-omit-frame-pointer)
else
  flags+=(-O2)
fi
"$compiler" "${flags[@]}" "${sources[@]}" -o "$library"
echo "$library"
