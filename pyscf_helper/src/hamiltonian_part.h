#pragma once
#include "hamiltonian.h"
#include "onstate.h"
#include "utils.h"

namespace squant {
using ham_tuple = std::tuple<float, float, float>;

ham_tuple get_Hij_part(const unsigned long *bra, const unsigned long *ket,
                       const double *h1e, const double *h2e, const size_t sorb,
                       const int nele, const int bra_len);
}  // namespace squant