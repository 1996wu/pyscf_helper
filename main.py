import os
import tempfile
import time
import numpy as np
import scipy.linalg

os.environ["OMP_NUM_THREADS"] = "4"

from pyscf_helper import read_integral, interface
from pyscf_helper.libs import get_hij, get_hij_part, get_comb, sparse_hij, sparse_hij_part

atom = ""
bond = 1.60
for k in range(10):
    atom += f"H, 0.00, {0.00}, {k * bond};"
integral_file = tempfile.mkstemp()[1]
sorb, nele, e_lst, fci_amp, ucisd_amp, mf = interface(
    atom,
    integral_file=integral_file,
    cisd_coeff=True,
    basis="STO-3G",
    # unit="", # bohr
    localized_orb=True,
    localized_method="meta-lowdin",
)

h1e, h2e, ci_space, ecore, sorb = read_integral(
    integral_file,
    nele,
)
noA = nele//2
noB = nele - noA

import scipy
from scipy.sparse import csr_matrix

# sparse matrix
t1 = time.time_ns()
data, cols, rows = sparse_hij(h1e, h2e, ci_space, sorb, nele, noA, noB)
# data_s, data_d, data_sign= sparse_hij_part(h1e, h2e, ci_space, sorb, nele, noA, noB)[:3]
# assert(np.allclose((np.asarray(data_s) + np.asarray(data_d)) * np.asarray(data_sign), np.asarray(data)))
data = np.asarray(data)
cols = np.asarray(cols)
rows = np.asarray(rows)
nbatch = ci_space.shape[0]
x = csr_matrix((data.flatten(), (rows.flatten(), cols.flatten())), shape=(nbatch, nbatch))
t2 = time.time_ns()
print(f"Sparse-matrix: {(t2 - t1)/1e06:.3f} ms")

print(scipy.sparse.linalg.eigsh(x)[0][0] + ecore, e_lst[0])

# Dense matrix
t1 = time.time_ns()
hij = get_hij(ci_space, ci_space, h1e, h2e, sorb, nele)
t2 = time.time_ns()
print(f"Dense-matrix: {(t2 - t1)/1e06:.3f} ms")
print(scipy.linalg.eigh(hij)[0][0] + ecore, e_lst[0])

single, double, sign = get_hij_part(ci_space, ci_space, h1e, h2e, sorb, nele)
assert np.allclose((single + double) * sign, hij)