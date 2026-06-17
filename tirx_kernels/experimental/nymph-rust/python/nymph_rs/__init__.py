"""nymph_rs — Rust-backed nymph.

The IR types + validation come from the compiled Rust extension; the IRBuilder
and the kernels are Python on top of it. `import nymph_rs` exposes all three.
"""

from . import kernels
from .builder import IRBuilder
from .nymph_rs import *
