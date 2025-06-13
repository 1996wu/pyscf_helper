#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "excitation.h"
#include "hamiltonian.h"
#include "hamiltonian_part.h"

namespace py = pybind11;
using ONV = py::array_t<uint8_t>;
using array = py::array_t<double>;

auto get_Hij(ONV bra, ONV ket, array h1e, array h2e, int sorb, int nele) {
  auto bra_buf = bra.request();
  auto ket_buf = ket.request();
  auto h1e_buf = h1e.request();
  auto h2e_buf = h2e.request();
  const int ket_dim = ket_buf.ndim;

  assert(ket_dim == 2 || ket_dim == 3);

  const size_t bra_len = (sorb - 1) / 64 + 1;

  bool flag_eloc = (ket_dim == 3);
  size_t n, m;

  if (flag_eloc) {
    // bra: (n, bra_len), ket: (n, m, bra_len)
    n = bra_buf.shape[0];
    m = ket_buf.shape[1];
  } else {
    // bra: (n, bra_len), ket: (m, bra_len)
    n = bra_buf.shape[0];
    m = ket_buf.shape[0];
  }

  if (bra_buf.size == 0 || ket_buf.size == 0) {
    return array({n, m});
  }

  array Hmat({n, m});
  auto Hmat_buf = Hmat.request();

  const auto* bra_ptr = reinterpret_cast<const unsigned long*>(bra_buf.ptr);
  const auto* ket_ptr = reinterpret_cast<const unsigned long*>(ket_buf.ptr);
  const auto* h1e_ptr = static_cast<const double*>(h1e_buf.ptr);
  const auto* h2e_ptr = static_cast<const double*>(h2e_buf.ptr);
  auto* Hmat_ptr = static_cast<double*>(Hmat_buf.ptr);

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t offset =
          flag_eloc ? (i * m * bra_len + j * bra_len) : (j * bra_len);
      Hmat_ptr[i * m + j] =
          squant::get_Hij_cpu(&bra_ptr[i * bra_len], &ket_ptr[offset], h1e_ptr,
                              h2e_ptr, sorb, nele, bra_len);
    }
  }

  return Hmat;
}

auto get_Hij_part(ONV bra, ONV ket, array h1e, array h2e, int sorb, int nele) {
  auto bra_buf = bra.request();
  auto ket_buf = ket.request();
  auto h1e_buf = h1e.request();
  auto h2e_buf = h2e.request();
  const int ket_dim = ket_buf.ndim;

  assert(ket_dim == 2 || ket_dim == 3);

  const size_t bra_len = (sorb - 1) / 64 + 1;

  bool flag_eloc = (ket_dim == 3);
  size_t n, m;

  if (flag_eloc) {
    // bra: (n, bra_len), ket: (n, m, bra_len)
    n = bra_buf.shape[0];
    m = ket_buf.shape[1];
  } else {
    // bra: (n, bra_len), ket: (m, bra_len)
    n = bra_buf.shape[0];
    m = ket_buf.shape[0];
  }

  if (bra_buf.size == 0 || ket_buf.size == 0) {
    auto x = array({n, m});
    return std::make_tuple(x, x, x);
  }

  array Hmat_s({n, m});
  array Hmat_d({n, m});
  array sign({n, m});
  auto Hmat_s_buf = Hmat_s.request();
  auto Hmat_d_buf = Hmat_d.request();
  auto sign_buf = sign.request();

  const auto* bra_ptr = reinterpret_cast<const unsigned long*>(bra_buf.ptr);
  const auto* ket_ptr = reinterpret_cast<const unsigned long*>(ket_buf.ptr);
  const auto* h1e_ptr = static_cast<const double*>(h1e_buf.ptr);
  const auto* h2e_ptr = static_cast<const double*>(h2e_buf.ptr);
  auto* Hmat_s_ptr = static_cast<double*>(Hmat_s_buf.ptr);
  auto* Hmat_d_ptr = static_cast<double*>(Hmat_d_buf.ptr);
  auto* sign_ptr = static_cast<double*>(sign_buf.ptr);

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      size_t offset =
          flag_eloc ? (i * m * bra_len + j * bra_len) : (j * bra_len);
      auto [x, y, z] =
          squant::get_Hij_part(&bra_ptr[i * bra_len], &ket_ptr[offset], h1e_ptr,
                               h2e_ptr, sorb, nele, bra_len);
      Hmat_s_ptr[i * m + j] = x;
      Hmat_d_ptr[i * m + j] = y;
      sign_ptr[i * m + j] = z;
    }
  }

  return std::make_tuple(Hmat_s, Hmat_d, sign);
}

auto tensor_to_onv(const ONV& bra_tensor, const int sorb) {
  // uint8: [1, 1, 0, 0] -> 0b0011 uint8
  py::buffer_info buf_info = bra_tensor.request();
  const auto* bra_ptr = static_cast<uint8_t*>(buf_info.ptr);
  int nbatch = buf_info.shape[0];
  const int bra_len = (sorb - 1) / 64 + 1;

  ONV states({nbatch, bra_len * 8});
  auto* states_ptr = reinterpret_cast<unsigned long*>(states.request().ptr);
  std::fill(states_ptr, states_ptr + nbatch * bra_len, 0);  // fill 0

#pragma omp parallel for
  for (int64_t i = 0; i < nbatch; i++) {
    for (int64_t j = 0; j < sorb; j++) {
      if (bra_ptr[i * sorb + j] == 1) {  // 1: occupied 0: unoccupied
        BIT_FLIP(states_ptr[i * bra_len + j / 64], j % 64);
      }
    }
  }

  return states;
}

array onv_to_tensor(const ONV& bra, const int sorb) {
  py::buffer_info bra_buf = bra.request();
  const auto* bra_ptr = reinterpret_cast<const unsigned long*>(bra_buf.ptr);
  auto shape = bra_buf.shape;
  int nbatch = shape[0];
  const int bra_len = (sorb - 1) / 64 + 1;

  array comb_bit({nbatch, sorb});
  auto* comb_ptr = static_cast<double*>(comb_bit.request().ptr);

#pragma omp parallel for
  for (int i = 0; i < nbatch; ++i) {
    squant::get_zvec_cpu(&bra_ptr[i * bra_len], &comb_ptr[i * sorb], sorb,
                         bra_len);
  }

  return comb_bit;
}

auto get_merged(const py::array_t<uint8_t>& bra, const int nele, const int sorb,
                const int noA, const int noB) {
  auto buf_info = bra.request();
  const int nbatch = buf_info.shape[0];
  const int bra_len = (sorb - 1) / 64 + 1;

  std::vector<int> merged(nbatch * sorb, 0);
  auto bra_ptr = reinterpret_cast<unsigned long*>(buf_info.ptr);

#pragma omp parallel for
  for (int i = 0; i < nbatch; ++i) {
    squant::get_olst_vlst_ab_cpu(&bra_ptr[i * bra_len], &merged[i * sorb], sorb,
                                 bra_len);
  }

  return merged;
}

auto get_comb(const ONV& bra, const int sorb, const int nele, const int noA,
              const int noB, bool flag_bit) {
  auto bra_buf = bra.request();
  const auto* bra_ptr = reinterpret_cast<const unsigned long*>(bra_buf.ptr);
  const int bra_len = (sorb - 1) / 64 + 1;
  const int ncomb = squant::get_Num_SinglesDoubles(sorb, noA, noB) + 1;
  const int nbatch = bra.shape(0);

  std::vector<unsigned long> comb(nbatch * ncomb * bra_len, 0);
#pragma omp parallel for
  for (int64_t i = 0; i < nbatch; i++) {
    for (int64_t j = 0; j < ncomb; j++) {
      std::memcpy(&comb[i * ncomb * bra_len + j * bra_len],
                  &bra_ptr[i * bra_len], bra_len * sizeof(unsigned long));
    }
  }
  auto merged = get_merged(bra, nele, sorb, noA, noB);  // nbatch * sorb

#pragma omp parallel for
  for (int64_t i = 0; i < nbatch; i++) {
    for (int64_t j = 1; j < ncomb; j++) {
      squant::get_comb_SD(&comb[i * ncomb * bra_len + j * bra_len],
                          &merged[i * sorb], j - 1, sorb, bra_len, noA, noB);
    }
  }
  auto comb_array = tools::as_pyarray(std::move(comb))
                        .reshape({nbatch, ncomb, bra_len})
                        .view("uint8");

  return comb_array;
}

auto sparse_hij(const array h1e, const array h2e, const ONV& bra,
                const int sorb, const int nele, const int noA, const int noB) {
  py::buffer_info bra_buf = bra.request();
  const auto* bra_ptr = reinterpret_cast<const unsigned long*>(bra_buf.ptr);
  const int bra_len = (sorb - 1) / 64 + 1;
  const int nSD = squant::get_Num_SinglesDoubles(sorb, noA, noB) + 1;
  const int nbatch = bra.shape(0);
  if (bra_len > 1) {
    throw std::overflow_error("sorb > 64");
  }
  std::vector<unsigned long> vec(bra_ptr, bra_ptr + nbatch);

  std::unordered_map<unsigned long, int> onv_Map;
  for (int64_t i = 0; i < nbatch; i++) {
    onv_Map[vec[i]] = i;
  }

  std::vector<double> hij_sparse(nbatch * nSD, 0.0);
  std::vector<int> cols(nbatch * nSD, 0);
  std::vector<int> rows(nbatch * nSD, 0);
  auto merged = get_merged(bra, nele, sorb, noA, noB);  // nbatch * sorb
  auto h1e_buf = h1e.request();
  auto h2e_buf = h2e.request();
  const auto* h1e_ptr = static_cast<const double*>(h1e_buf.ptr);
  const auto* h2e_ptr = static_cast<const double*>(h2e_buf.ptr);

#pragma omp parallel for
  for (int64_t i = 0; i < nbatch; i++) {
    std::vector<unsigned long> SD_comb(nSD, bra_ptr[i * bra_len]);
    std::fill_n(&cols[i * nSD], nSD, i);
    for (int64_t j = 0; j < nSD; j++) {
      if (j >= 1) {
        squant::get_comb_SD(&SD_comb[j * bra_len], &merged[i * sorb], j - 1,
                            sorb, bra_len, noA, noB);
      }
      hij_sparse[i * nSD + j] =
          squant::get_Hij_cpu(&bra_ptr[i * bra_len], &SD_comb[j * bra_len],
                              h1e_ptr, h2e_ptr, sorb, nele, bra_len);
      rows[i * nSD + j] = onv_Map.find(SD_comb[j * bra_len]) != onv_Map.end()
                              ? onv_Map[SD_comb[j * bra_len]]
                              : -1;
      // if (rows[i * nSD + j] == -1){
      //     exit(0);
      //   }
    }
  }

  // std::vector<std::vector<int>> rows_view(nbatch);
  // std::vector<std::vector<double>> hij_sparse_view(nbatch);

  // for (int i = 0; i < nbatch; ++i) {
  //   rows_view[i] =
  //       std::vector<int>(rows.begin() + i * nSD, rows.begin() + (i + 1) *
  //       nSD);
  // }

  return std::make_tuple(hij_sparse, cols, rows);
}

auto sparse_hij_part(const array h1e, const array h2e, const ONV& bra,
                     const int sorb, const int nele, const int noA,
                     const int noB) {
  py::buffer_info bra_buf = bra.request();
  const auto* bra_ptr = reinterpret_cast<const unsigned long*>(bra_buf.ptr);
  const int bra_len = (sorb - 1) / 64 + 1;
  const int nSD = squant::get_Num_SinglesDoubles(sorb, noA, noB) + 1;
  const int nbatch = bra.shape(0);
  if (bra_len > 1) {
    throw std::overflow_error("sorb > 64");
  }
  std::vector<unsigned long> vec(bra_ptr, bra_ptr + nbatch);

  std::unordered_map<unsigned long, int> onv_Map;
  for (int64_t i = 0; i < nbatch; i++) {
    onv_Map[vec[i]] = i;
  }

  std::vector<double> hij_sparse_s(nbatch * nSD, 0.0);
  std::vector<double> hij_sparse_d(nbatch * nSD, 0.0);
  std::vector<double> hij_sparse_sign(nbatch * nSD, 0.0);
  std::vector<int> cols(nbatch * nSD, 0);
  std::vector<int> rows(nbatch * nSD, 0);
  auto merged = get_merged(bra, nele, sorb, noA, noB);  // nbatch * sorb
  auto h1e_buf = h1e.request();
  auto h2e_buf = h2e.request();
  const auto* h1e_ptr = static_cast<const double*>(h1e_buf.ptr);
  const auto* h2e_ptr = static_cast<const double*>(h2e_buf.ptr);

#pragma omp parallel for
  for (int64_t i = 0; i < nbatch; i++) {
    std::vector<unsigned long> SD_comb(nSD, bra_ptr[i * bra_len]);
    std::fill_n(&cols[i * nSD], nSD, i);
    for (int64_t j = 0; j < nSD; j++) {
      if (j >= 1) {
        squant::get_comb_SD(&SD_comb[j * bra_len], &merged[i * sorb], j - 1,
                            sorb, bra_len, noA, noB);
      }
      auto [x, y, z] =
          squant::get_Hij_part(&bra_ptr[i * bra_len], &SD_comb[j * bra_len],
                               h1e_ptr, h2e_ptr, sorb, nele, bra_len);
      hij_sparse_s[i * nSD + j] = x;
      hij_sparse_d[i * nSD + j] = y;
      hij_sparse_sign[i * nSD + j] = z;
      rows[i * nSD + j] = onv_Map.find(SD_comb[j * bra_len]) != onv_Map.end()
                              ? onv_Map[SD_comb[j * bra_len]]
                              : -1;
    }
  }
  return std::make_tuple(hij_sparse_s, hij_sparse_d, hij_sparse_sign, cols,
                         rows);
}

PYBIND11_MODULE(libs, m) {
  m.def("get_hij", &get_Hij, py::arg("bra"), py::arg("ket"), py::arg("h1e"),
        py::arg("h2e"), py::arg("sorb"), py::arg("nele"),
        "Calculate the matrix <x|H|x'> using CPU with numpy");
  m.def("get_hij_part", &get_Hij_part, py::arg("bra"), py::arg("ket"),
        py::arg("h1e"), py::arg("h2e"), py::arg("sorb"), py::arg("nele"),
        "Calculate the matrix <x|H|x'> using CPU with numpy");
  m.def("tensor_to_onv", &tensor_to_onv, py::arg("bra"), py::arg("sorb"),
        "convert states (0:unoccupied, 1: occupied) to onv uint8");
  m.def("onv_to_tensor", &onv_to_tensor, py::arg("bra"), py::arg("sorb"),
        "convert onv to bit (-1:unoccupied, 1: occupied) for given onv "
        "using CPU with numpy");
  m.def("get_comb", &get_comb, py::arg("bra"), py::arg("sorb"), py::arg("nele"),
        py::arg("noA"), py::arg("noB"), py::arg("flag_bit") = false,
        "Return all singles and doubles excitation for given onv"
        "using CPU with numpy");
  m.def("sparse_hij", &sparse_hij);
  m.def("sparse_hij_part", &sparse_hij_part);
}
