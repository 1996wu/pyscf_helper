import numpy as np
from numpy import ndarray
from pyscf_helper.libs import get_hij


def energy_CI(
    coeff: ndarray,
    onstate: ndarray,
    h1e: ndarray,
    h2e: ndarray,
    ecore: float,
    sorb: int,
    nele: int,
    batch: int = -1,
) -> float:
    """
    e = <psi|H|psi>/<psi|psi>
      <psi|H|psi> = \sum_{ij}c_i<i|H|j>c_j*
    """
    assert coeff.shape[0] == onstate.shape[0]
    dim = onstate.shape[0]
    if batch == -1:
        batch = dim
    else:
        assert batch > 0

    chunks_onv = np.array_split(onstate, int((dim - 1) / batch) + 1, axis=0)
    chunks_ci = np.array_split(coeff, int((dim - 1) / batch) + 1, axis=0)
    chunks_e = np.zeros((len(chunks_onv), len(chunks_onv)), dtype=np.complex128)

    for i in range(len(chunks_onv)):
        p1_onv, p1_ci = chunks_onv[i], chunks_ci[i]
        for j in range(i, len(chunks_onv)):
            p2_onv, p2_ci = chunks_onv[j], chunks_ci[j]
            hij = get_hij(p1_onv, p2_onv, h1e, h2e, sorb, nele)
            e = np.einsum("i, ij, j", p1_ci.flatten(), hij, np.conj(p2_ci.flatten()))
            chunks_e[i, j] = chunks_e[j, i] = e

    # Normalize and add the ecore energy
    e = chunks_e.sum() / np.linalg.norm(coeff) ** 2 + ecore
    return e