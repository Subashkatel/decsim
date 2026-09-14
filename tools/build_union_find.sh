#!/usr/bin/env bash
# The compiled Union-Find decoder, built into the package folder beside
# its source. The plain build is what decsim loads; the sanitized build
# is a second library the corpus runs under to prove the C stays inside
# its buffers, and it is never the one a run uses.
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
source_file=decsim/decoders/union_find/union_find.c
flags=(-std=c11 -Wall -Wextra -Wpedantic -Werror -fPIC -shared)
library=decsim/decoders/union_find/union_find.so
if [ "${1:-}" = "--sanitize" ]; then
  library=decsim/decoders/union_find/union_find_sanitized.so
  flags+=(-O1 -g -fsanitize=address,undefined)
  flags+=(-fsanitize-undefined-trap-on-error -fno-omit-frame-pointer)
else
  flags+=(-O2)
fi
"$compiler" "${flags[@]}" "$source_file" -o "$library"
echo "$library"
