#!/usr/bin/env bash
# Build the nymph_rs extension and run the Python binding/integration tests.
#
# site-packages is read-only here, so instead of `maturin develop` we build a
# wheel, unzip it into ./_pybuild, and put that on PYTHONPATH.
#
# PYO3_USE_ABI3_FORWARD_COMPATIBILITY lets PyO3 build the abi3 wheel against a
# Python newer than it officially lists (e.g. 3.14); harmless on older versions.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin build --release
rm -rf _pybuild && mkdir -p _pybuild
python -m zipfile -e target/wheels/nymph_rs-*.whl _pybuild/

PYTHONPATH="$PWD/_pybuild" python -m pytest tests/ "$@"
