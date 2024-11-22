#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

#include "default.h"

namespace tools {

template <typename T>
double get_duration_nano(T t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

inline std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}

// ref: https://github.com/ssciwr/pybind11-numpy-example/ move
template <typename Sequence>
inline pybind11::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq) {
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr =
      std::make_unique<Sequence>(std::move(seq));
  auto capsule = pybind11::capsule(seq_ptr.get(), [](void *p) {
    std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p));
  });
  seq_ptr.release();
  return pybind11::array(size, data, capsule);
}


}
