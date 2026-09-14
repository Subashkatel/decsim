"""What the suite needs built before any test runs.

The Union-Find row decodes in C, and the library it loads is a build
artifact rather than a tracked file, so a fresh checkout with a
compiler has none. One session-wide fixture builds it, and a checkout
without a compiler fails here with the build command named rather than
inside every test that decodes.
"""

import pathlib
import subprocess

import pytest

import decsim.decoders.union_find.compiled_decoder as compiled_decoder

TESTS_FILE = pathlib.Path(__file__)
TESTS_PATH = TESTS_FILE.resolve()
CHECKOUT = TESTS_PATH.parent.parent


@pytest.fixture(scope="session", autouse=True)
def compiled_union_find_library():
    """The compiled Union-Find decoder, built once when it is missing."""
    path = compiled_decoder.library_path()
    if path.exists():
        return path
    command = CHECKOUT / compiled_decoder.BUILD_COMMAND
    subprocess.run([str(command)], cwd=CHECKOUT, check=True)
    return path
