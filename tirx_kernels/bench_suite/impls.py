"""Implementation-name classification shared by bench-suite reports."""

from collections.abc import Mapping


def is_our_impl(name: str) -> bool:
    """Whether ``name`` identifies a TIR/TIRx implementation."""
    return name in {"tir", "tirx"} or name.startswith(("tir_", "tirx_"))


def our_impls(impls: Mapping[str, float]) -> list[str]:
    """Return all TIR/TIRx implementation names in result order."""
    return [name for name in impls if is_our_impl(name)]
