"""What the suite needs built before any test runs.

The Union-Find row decodes in C, and the library it loads is a build
artifact rather than a tracked file, so a fresh checkout has none. One
session-wide fixture builds it where a compiler is reachable. Where one
is not, or where the build fails, the session stops with the sentence
that names the command, because the alternative is the same build error
retold once per test.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

import decsim.decoders.union_find.compiled_decoder as compiled_decoder

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
CHECKOUT = TESTS_PATH.parent.parent
BUILD_SENTENCE = (
    "the Union-Find decoder needs its compiled library: run "
    f"{compiled_decoder.BUILD_COMMAND} on a host with a C compiler, then "
    "run the suite again"
)


@pytest.fixture(scope="session", autouse=True)
def compiled_union_find_library():
    """The compiled Union-Find decoder, built once when it is missing."""
    path = compiled_decoder.library_path()
    if path.exists():
        return path
    compiler = os.environ.get("CC", "gcc")
    if shutil.which(compiler) is None:
        pytest.exit(BUILD_SENTENCE, returncode=1)
    command = CHECKOUT / compiled_decoder.BUILD_COMMAND
    build = subprocess.run([str(command)], cwd=CHECKOUT, check=False)
    if build.returncode != 0:
        pytest.exit(BUILD_SENTENCE, returncode=1)
    return path
