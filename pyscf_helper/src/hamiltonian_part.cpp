#include "hamiltonian_part.h"

namespace squant {

ham_tuple get_Hii_part(const unsigned long *bra, const unsigned long *ket,
                       const double *h1e, const double *h2e, const int sorb,
                       const int nele, const int bra_len) {
  double Hij_s = 0.0, Hij_d = 0.0;
  int olst[MAX_NELE] = {0};
  get_olst_cpu(bra, olst, bra_len);

  for (int i = 0; i < nele; i++) {
    int p = olst[i];  //<p|h|p>
    Hij_s += h1e_get_cpu(h1e, p, p, sorb);
    for (int j = 0; j < i; j++) {
      int q = olst[j];
      Hij_d += h2e_get_cpu(h2e, p, q, p, q);  //<pq||pq> Storage not continuous
    }
  }
  return std::make_tuple(Hij_s, Hij_d, 1.0);
}

ham_tuple get_HijS_part(const unsigned long *bra, const unsigned long *ket,
                        const double *h1e, const double *h2e, const size_t sorb,
                        const int bra_len) {
  double Hij_s = 0.0, Hij_d = 0.0;
  int p[1], q[1];
  diff_orb_cpu(bra, ket, bra_len, p, q);
  Hij_s += h1e_get_cpu(h1e, p[0], q[0], sorb);  // hpq
  for (int i = 0; i < bra_len; i++) {
    unsigned long repr = bra[i];
    while (repr != 0) {
      int j = 63 - __builtin_clzl(repr);
      int k = 64 * i + j;
      Hij_d += h2e_get_cpu(h2e, p[0], k, q[0], k);  //<pk||qk>
      repr &= ~(1ULL << j);
    }
  }
  auto sgn = static_cast<double>(parity_cpu(bra, p[0]) * parity_cpu(ket, q[0]));
  return std::make_tuple(Hij_s, Hij_d, sgn);
}

ham_tuple get_HijD_part(const unsigned long *bra, const unsigned long *ket,
                        const double *h1e, const double *h2e, const size_t sorb,
                        const int bra_len) {
  int p[2], q[2];
  diff_orb_cpu(bra, ket, bra_len, p, q);
  int sgn = parity_cpu(bra, p[0]) * parity_cpu(bra, p[1]) *
            parity_cpu(ket, q[0]) * parity_cpu(ket, q[1]);
  double Hij_d = h2e_get_cpu(h2e, p[0], p[1], q[0], q[1]);
  return std::make_tuple(0.0, Hij_d, static_cast<double>(sgn));
}

ham_tuple get_Hij_part(const unsigned long *bra, const unsigned long *ket,
                       const double *h1e, const double *h2e, const size_t sorb,
                       const int nele, const int bra_len) {
  ham_tuple part = std::make_tuple(0.0, 0.0, 1.0);
  int type[2] = {0};
  diff_type_cpu(bra, ket, type, bra_len);
  if (type[0] == 0 && type[1] == 0) {
    // std::cout << "Hii" << std::endl;
    part = get_Hii_part(bra, ket, h1e, h2e, sorb, nele, bra_len);
  } else if (type[0] == 1 && type[1] == 1) {
    part = get_HijS_part(bra, ket, h1e, h2e, sorb, bra_len);
  } else if (type[0] == 2 && type[1] == 2) {
    // std::cout << "Hij_D" << std::endl;
    part = get_HijD_part(bra, ket, h1e, h2e, sorb, bra_len);
  }
  return part;
}

}  // namespace squant