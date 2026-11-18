"""NumPy CPU bindings for integrals, Slater--Condon elements and strings Hv.

ONVs use interleaved alpha/beta occupations, little-endian bits and complete
64-bit words: uint8 (..., 8*ceil(sorb/64)). Integrals and CI vectors use FP64.
"""

from typing import TypeAlias, TypedDict
import numpy as np
from numpy.typing import NDArray

_ONV: TypeAlias = NDArray[np.uint8]
_FloatArray: TypeAlias = NDArray[np.float64]


class _HamiltonianInfo(TypedDict):
    n_alpha_strings: int
    n_beta_strings: int
    alpha_single_links: int
    beta_single_links: int
    alpha_links_cached: bool
    beta_links_cached: bool
    alpha_double_links_cached: bool
    beta_double_links_cached: bool
    alpha_double_links: int
    beta_double_links: int
    parallel_runtime: str
    threads: int
    openmp_dynamic: bool
    openmp_schedule: str
    parity_bytes: int
    onv_words: int
    spin_string_bits: int
    mixed_lookup_workspace_bytes: int
    offdiagonal_value_cache_bytes: int
    connectivity_bytes: int


class _ProfileResult(TypedDict):
    sampled_rows: int
    sc_repeats: int
    wall_s: float
    phase_s: dict[str, float]
    connections: dict[str, int]
    original_slater_condon_s: tuple[float, float, float]
    slater_condon_max_abs_error: float
    rows: list[int]
    sigma: list[float]


class _StringHamiltonian64:
    def diagonal(self) -> _FloatArray: ...
    def matvec(self, v: _FloatArray, /) -> _FloatArray: ...
    def profile(self, v: _FloatArray, rows: int = 512, repeats: int = 3) -> _ProfileResult: ...
    def info(self) -> _HamiltonianInfo: ...


class _StringHamiltonian128:
    def diagonal(self) -> _FloatArray: ...
    def matvec(self, v: _FloatArray, /) -> _FloatArray: ...
    def profile(self, v: _FloatArray, rows: int = 512, repeats: int = 3) -> _ProfileResult: ...
    def info(self) -> _HamiltonianInfo: ...


def StringHamiltonian(
    onv: _ONV,
    h1e: _FloatArray,
    h2e: _FloatArray,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    link_budget: int,
    /,
) -> _StringHamiltonian64 | _StringHamiltonian128:
    """Factory for one fixed-space operator; key width is selected internally.

    Inputs must be contiguous. h1e/h2e are compressed one-dimensional integrals;
    link_budget is the string-adjacency cache budget in bytes. The operator does
    not cache off-diagonal matrix elements. Supported even sorb is at most 256.
    """
    ...


def get_hij(
    bra: _ONV, ket: _ONV, h1e: _FloatArray, h2e: _FloatArray,
    sorb: int, nele: int,
) -> _FloatArray:
    """SC elements for ket (M, width) or (N, M, width); result is (N, M)."""
    ...


def get_hij_part(
    bra: _ONV, ket: _ONV, h1e: _FloatArray, h2e: _FloatArray,
    sorb: int, nele: int,
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Three SC-part arrays, each with the same shape as get_hij output."""
    ...


def get_comb(
    bra: _ONV, sorb: int, nele: int, noA: int, noB: int,
    flag_bit: bool = False,
) -> _ONV:
    """Reference plus all spin-conserving single/double excitations, packed.

    Output is (N, n_SD+1, width). flag_bit is retained by the binding but the
    current implementation always returns packed ONVs.
    """
    ...


def sparse_hij(
    h1e: _FloatArray, h2e: _FloatArray, bra: _ONV,
    sorb: int, nele: int, noA: int, noB: int, /,
) -> tuple[list[float], list[int], list[int]]:
    """Legacy flat element/column/row lists; absent target indices are -1.

    This separate helper supports sorb <= 64; fixed-space strings Hv does not
    call it or build these arrays.
    """
    ...


def sparse_hij_part(
    h1e: _FloatArray, h2e: _FloatArray, bra: _ONV,
    sorb: int, nele: int, noA: int, noB: int, /,
) -> tuple[list[float], list[float], list[float], list[int], list[int]]:
    """Legacy flat SC-part, column and row lists; sorb <= 64."""
    ...


def compress_h1e_h2e(
    h1e: _FloatArray, h2e: _FloatArray, sorb: int,
) -> tuple[_FloatArray, _FloatArray]:
    """(sorb,sorb)/(sorb,)*4 -> (sorb**2,)/(pair*(pair+1)//2,)."""
    ...


def decompress_h1e_h2e(
    h1e: _FloatArray, h2e: _FloatArray, sorb: int,
) -> tuple[_FloatArray, _FloatArray]:
    """Inverse of compress_h1e_h2e for the existing antisymmetrized convention."""
    ...
