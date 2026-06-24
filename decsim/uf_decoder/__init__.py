"""Base Union-Find decoder of the Coset Ensemble Decoder (arXiv:2606.11076).

PORTED third-party code -- NOT decsim's own authorship. ``software/uf_original.py`` and
``tools/post_precessing.py`` are the authors' real decoder; ``code_structure.py`` is the
``CodeStructure`` input class the decoder requires. See README.md for provenance, the exact trims
made, and license status (upstream ships no license -> research-reproduction use with attribution).

The only file written by us is this ``__init__.py``. The upstream files use absolute imports
(``from software.uf_original import ...``, ``from tools.post_precessing import ...``) and assume the
package root is on ``sys.path``, so we add this directory to ``sys.path`` (side effect: the
top-level names ``software`` and ``tools`` become importable process-wide).

Public API:
  * ``uf_original(syndrome, code_structure, error_type='x', grow_mode='parallel')`` -> full decode
    (Union-Find clustering + peeling); returns ``(corrections, weights)``.
  * ``union_find_decoder(...)`` -> the clustering stage alone.
  * ``CodeStructure(H_x, H_z, logicals_x, logicals_z, L, repetitions=...)`` -> the decoder's input.
"""
import os as _os
import sys as _sys

_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _PKG_DIR not in _sys.path:
    _sys.path.insert(0, _PKG_DIR)

from software.uf_original import uf_original, union_find_decoder  # noqa: E402
from .code_structure import CodeStructure

__all__ = ["uf_original", "union_find_decoder", "CodeStructure"]
