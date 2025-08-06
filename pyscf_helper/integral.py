from __future__ import annotations

import time
import os
import warnings
import itertools
import numpy as np
import sys
sys.path.append("./")

from numpy import ndarray
from typing import Tuple

from pyscf_helper.libs import tensor_to_onv
__all__ = ["read_integral", "Integral"]


def get_special_space(x: int, sorb: int, noa: int, nob: int) -> ndarray:
    """
    Generate all or part of FCI-state
    """
    assert x % 2 == 0 and x <= sorb and x >= (noa + nob)
    # the order is different from pyscf.fci.cistring._gen_occslst(iterable, r)
    # the 'gen_occslst' is pretty slow than 'combinations', and only is used exact optimization testing.
    if x == sorb:
        from pyscf import fci

        noA_lst = fci.cistring.gen_occslst([i for i in range(0, x, 2)], noa)
        noB_lst = fci.cistring.gen_occslst([i for i in range(1, x, 2)], nob)
    else:
        noA_lst = list(itertools.combinations([i for i in range(0, x, 2)], noa))
        noB_lst = list(itertools.combinations([i for i in range(1, x, 2)], nob))

    m = len(noA_lst)
    n = len(noB_lst)
    spins = np.zeros((m * n, sorb), dtype=np.uint8)
    for i, lstA in enumerate(noA_lst):
        for j, lstB in enumerate(noB_lst):
            idx = i * m + j
            spins[idx, lstA] = 1
            spins[idx, lstB] = 1

    return tensor_to_onv(spins, sorb)
    # return convert_onv(spins, sorb=sorb, device=device)

class Integral:
    """
    Real int1e and int2e from integral file with pyscf interface,
    """

    def __init__(self, filename: str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Integral file{filename}  dose not exits")
        self.file = filename
        self.sorb: int = 0

    def init_data(self):
        self.pair = self.sorb * (self.sorb - 1) // 2
        self.int1e = np.zeros((self.sorb * self.sorb), dtype=np.float64)  # <i|O1|j>
        self.int2e = np.zeros((self.pair * (self.pair + 1)) // 2, dtype=np.float64)  # <ij||kl>
        self.ecore = 0.00
        self._information()

    def _one_body(self, i: int, j: int, value: float):
        self.int1e[i * self.sorb + j] = value

    def _two_body(self, i: int, j: int, k: int, l: int, value: float):
        if (i == j) or (k == l):
            return
        ij = (i * (i - 1)) // 2 + j if i > j else (j * (j - 1)) // 2 + i
        kl = (k * (k - 1)) // 2 + l if k > l else (l * (l - 1)) // 2 + k
        sgn = 1.00
        sgn = sgn if i > j else -1 * sgn
        sgn = sgn if k > l else -1 * sgn
        if ij >= kl:
            ijkl = (ij * (ij + 1)) // 2 + kl
            self.int2e[ijkl] = sgn * value
        else:
            ijkl = (kl * (kl + 1)) // 2 + ij
            self.int2e[ijkl] = sgn * value.conjugate()

    def load(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        t0 = time.time_ns()
        with open(self.file, "r") as f:
            for a, lines in enumerate(f):
                if a == 0:
                    self.sorb = int(lines.split()[0])
                    self.init_data()
                else:
                    line = lines.split()
                    i, j, k, l = tuple(map(int, line[:-1]))
                    value = float(line[-1])
                    if (i * j == 0) and (k * l == 0):
                        self.ecore = value
                    elif (i * j != 0) and (k * l == 0):
                        self._one_body(i - 1, j - 1, value)
                    elif (i * j != 0) and (k * l != 0):
                        self._two_body(i - 1, j - 1, k - 1, l - 1, value)
        print(f"Time for loading integral: {(time.time_ns() - t0)/1.0E09:.3E}s")

        return self.int1e, self.int2e, self.ecore, self.sorb

    def _information(self):
        int2e_mem = self.int2e.shape[0] * 8 / (1 << 20)  # MiB
        int1e_mem = self.int1e.shape[0] * 8 / (1 << 20)  # MiB
        s = f"Integral file: {self.file}\n"
        s += f"Sorb: {self.sorb}\n"
        s += f"int1e: {self.int1e.shape[0]}:{int1e_mem:.3E}MiB\n"
        s += f"int2e: {self.int2e.shape[0]}:{int2e_mem:.3E}MiB"
        print(s)


def read_integral(
    filename: str,
    nele: int,
    given_sorb: int = None,
) -> Tuple[ndarray, ndarray, ndarray, float, int]:
    """
    read the int2e, int1e, ecore for integral file
    dose not support Sz != 0, and will implement in future.

    Returns:
    -------
        h1e, h2e: np.float64
        onstate: np.uint8 in Full-CI space or part-FCI space
        ecore: float
        sorb: int
    """

    t = Integral(filename)
    h1e, h2e, ecore, sorb = t.load()

    noa = nele // 2
    nob = nele // 2
    if given_sorb is None:
        given_sorb = sorb
    onstate = get_special_space(given_sorb, sorb, noa, nob)
    if given_sorb != sorb:
        import warnings
        from scipy import special

        n1 = special.comb(sorb // 2, noa, exact=True)
        n2 = special.comb(sorb // 2, nob, exact=True)
        fci_size = n1 * n2
        warnings.warn(
            f"The CI-space({onstate.size(0):.3E}) is part of the FCI-space({fci_size:.3E})"
        )

    return (h1e, h2e, onstate, ecore, sorb)