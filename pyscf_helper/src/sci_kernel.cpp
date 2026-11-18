#include "sci_kernel.h"

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "hamiltonian.h"
#include "onstate.h"

// Connectivity: Fast SHCI, arXiv:1809.04600, Section III / Algorithms 1--2.
// Matrix elements reuse the existing pyscf_helper Slater--Condon functions.
// Row ownership avoids atomics in sigma. No matrix-element threshold is used.
namespace py = pybind11;
namespace {
using Row = std::int32_t;

static unsigned long compact_even(unsigned long x) {
  x &= 0x5555555555555555ULL;
  x = (x | (x >> 1)) & 0x3333333333333333ULL;
  x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0fULL;
  x = (x | (x >> 4)) & 0x00ff00ff00ff00ffULL;
  x = (x | (x >> 8)) & 0x0000ffff0000ffffULL;
  return (x | (x >> 16)) & 0x00000000ffffffffULL;
}

template <class String>
struct StringHash {
  std::size_t operator()(String x) const {
    auto hash = std::hash<unsigned long>{}(static_cast<unsigned long>(x));
    if constexpr (sizeof(String) > sizeof(unsigned long))
      hash ^= std::hash<unsigned long>{}(static_cast<unsigned long>(x >> 64)) +
              0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
  }
};

template <class String>
static String orbital_mask(int norb) {
  return norb == sizeof(String) * 8 ? ~String{0} : (String{1} << norb) - 1;
}

template <class String>
static int popcount(String x) {
  int count = __builtin_popcountl(static_cast<unsigned long>(x));
  if constexpr (sizeof(String) > sizeof(unsigned long))
    count += __builtin_popcountl(static_cast<unsigned long>(x >> 64));
  return count;
}

template <class String>
struct Groups {
  std::vector<String> keys;
  std::vector<Row> rows, id, secondary_id;
  std::vector<std::size_t> ptr;
  std::vector<double> scan_weight;
};

template <class String>
static Groups<String> make_groups(const std::vector<String>& primary,
                                  const std::vector<String>& secondary) {
  Groups<String> g;
  g.rows.resize(primary.size());
  g.id.resize(primary.size());
  std::iota(g.rows.begin(), g.rows.end(), 0);
  std::sort(g.rows.begin(), g.rows.end(), [&](Row i, Row j) {
    return primary[i] != primary[j] ? primary[i] < primary[j]
                                    : secondary[i] < secondary[j];
  });
  for (std::size_t p = 0; p < g.rows.size(); ++p) {
    Row i = g.rows[p];
    if (p == 0 || primary[i] != g.keys.back()) {
      g.keys.push_back(primary[i]);
      g.ptr.push_back(p);
    }
    g.id[i] = static_cast<Row>(g.keys.size() - 1);
  }
  g.ptr.push_back(g.rows.size());
  for (std::size_t i = 0; i < g.keys.size(); ++i)
    g.scan_weight.push_back(1 + std::log2(g.ptr[i + 1] - g.ptr[i] + 1));
  return g;
}

struct Singles {
  std::vector<std::size_t> ptr;
  std::vector<Row> targets;
  bool cached = false;
};

// One-electron-deleted signatures, using holes when there are fewer holes.
// The memory guard falls back to exact on-demand string generation, never
// truncation.
template <class String>
static Singles make_singles(const std::vector<String>& keys, int norb,
                            int occupied, std::size_t budget) {
  Singles out;
  const int objects = std::min(occupied, norb - occupied);
  if (keys.size() * static_cast<double>(objects) * 64 > budget) return out;
  const String mask = orbital_mask<String>(norb);
  const bool holes = occupied > norb - occupied;
  std::unordered_map<String, std::vector<Row>, StringHash<String>> buckets;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    String x = holes ? keys[i] ^ mask : keys[i];
    for (String left = x; left; left &= left - 1) {
      String bit = left & -left;
      buckets[x ^ bit].push_back(static_cast<Row>(i));
    }
  }
  std::vector<std::vector<Row>> links(keys.size());
  std::size_t edges = 0;
  for (const auto& item : buckets) {
    const auto& ids = item.second;
    for (std::size_t p = 0; p < ids.size(); ++p)
      for (std::size_t q = p + 1; q < ids.size(); ++q) {
        if ((edges + 2) * sizeof(Row) * 3 + keys.size() * 32 > budget)
          return Singles{};
        links[ids[p]].push_back(ids[q]);
        links[ids[q]].push_back(ids[p]);
        edges += 2;
      }
  }
  out.ptr.push_back(0);
  for (auto& list : links) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    out.targets.insert(out.targets.end(), list.begin(), list.end());
    out.ptr.push_back(out.targets.size());
  }
  out.cached = true;
  return out;
}

// Same-spin double links are shared by every spectator-spin group. Store only
// string IDs, with two-pass allocation bounded by the remaining link budget.
template <class String>
static Singles make_doubles(const std::vector<String>& keys, int norb,
                            int occupied, std::size_t budget) {
  Singles out;
  if (keys.size() * 64.0 > budget) return out;
  std::unordered_map<String, Row, StringHash<String>> ids;
  ids.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i)
    ids.emplace(keys[i], static_cast<Row>(i));
  const String mask = orbital_mask<String>(norb);
  auto visit = [&](String x, auto emit) {
    std::vector<String> os, vs;
    for (String left = x; left; left &= left - 1) os.push_back(left & -left);
    for (String left = mask ^ x; left; left &= left - 1)
      vs.push_back(left & -left);
    for (std::size_t i = 0; i < os.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        for (std::size_t a = 0; a < vs.size(); ++a)
          for (std::size_t b = 0; b < a; ++b) {
            auto it = ids.find(x ^ os[i] ^ os[j] ^ vs[a] ^ vs[b]);
            if (it != ids.end()) emit(it->second);
          }
  };
  out.ptr.push_back(0);
  std::size_t edges = 0;
  for (String x : keys) {
    visit(x, [&](Row) { ++edges; });
    if (edges * sizeof(Row) + keys.size() * 64.0 > budget) return Singles{};
    out.ptr.push_back(edges);
  }
  out.targets.resize(edges);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    std::size_t p = out.ptr[i];
    visit(keys[i], [&](Row j) { out.targets[p++] = j; });
    std::sort(out.targets.begin() + out.ptr[i],
              out.targets.begin() + out.ptr[i + 1]);
  }
  out.cached = true;
  return out;
}

static std::size_t link_bytes(const Singles& links) {
  return links.ptr.size() * sizeof(std::size_t) +
         links.targets.size() * sizeof(Row);
}

template <class String>
static std::vector<std::size_t> scan_work(const Singles& links,
                                          const Groups<String>& group) {
  if (!links.cached) return {};
  std::vector<std::size_t> work(group.keys.size(), 0);
  for (std::size_t i = 0; i < work.size(); ++i)
    for (std::size_t p = links.ptr[i]; p < links.ptr[i + 1]; ++p) {
      Row j = links.targets[p];
      work[i] += group.ptr[j + 1] - group.ptr[j];
    }
  return work;
}


struct GroupBitmap {
  std::size_t stride = 0;
  std::vector<unsigned long> bits;
  std::vector<Row> rank;
};

template <class String>
static GroupBitmap group_bitmap(const Groups<String>& g, std::size_t columns) {
  GroupBitmap out;
  out.stride = (columns + 63) / 64;
  out.bits.assign(g.keys.size() * out.stride, 0);
  out.rank.resize(out.bits.size());
  for (std::size_t group = 0; group < g.keys.size(); ++group) {
    auto base = group * out.stride;
    for (auto p = g.ptr[group]; p < g.ptr[group + 1]; ++p) {
      auto id = static_cast<std::size_t>(g.secondary_id[p]);
      out.bits[base + id / 64] |= 1UL << (id % 64);
    }
    Row count = 0;
    for (std::size_t w = 0; w < out.stride; ++w) {
      out.rank[base + w] = count;
      count += __builtin_popcountl(out.bits[base + w]);
    }
  }
  return out;
}

struct NeighborWords {
  std::vector<std::size_t> ptr;
  std::vector<Row> word;
  std::vector<unsigned long> mask;
};

static NeighborWords neighbor_words(const Singles& graph) {
  NeighborWords out;
  out.ptr.push_back(0);
  for (std::size_t group = 0; group + 1 < graph.ptr.size(); ++group) {
    for (auto p = graph.ptr[group]; p < graph.ptr[group + 1]; ++p) {
      Row id = graph.targets[p], w = id / 64;
      if (out.word.size() == out.ptr.back() || out.word.back() != w) {
        out.word.push_back(w);
        out.mask.push_back(0);
      }
      out.mask.back() |= 1UL << (id % 64);
    }
    out.ptr.push_back(out.word.size());
  }
  return out;
}

template <class String>
class FixedSpaceHamiltonian {
  py::array_t<std::uint8_t, py::array::c_style> onv_owner;
  py::array_t<double, py::array::c_style> h1_owner, h2_owner;
  const unsigned long* onv;
  const double *h1, *h2;
  std::size_t n;
  int sorb, norb, nele, noa, nob, width;
  std::vector<String> alpha, beta;
  Groups<String> ga, gb;
  Singles sa, sb, da, db;
  GroupBitmap bga, bgb;
  NeighborWords bsa, bsb;
  std::vector<std::size_t> alpha_scan_work, beta_scan_work;
  std::vector<double> diag;
  std::vector<unsigned long> fermion_parity;
  // Per-thread neighbor membership scratch, bounded by the link budget.
  std::size_t mixed_marker_size = 0;

  std::vector<Row> generated_singles(String x, const Groups<String>& g) const {
    std::vector<Row> result;
    const String virtuals = (orbital_mask<String>(norb)) ^ x;
    for (String o = x; o; o &= o - 1)
      for (String v = virtuals; v; v &= v - 1) {
        String next = x ^ (o & -o) ^ (v & -v);
        auto it = std::lower_bound(g.keys.begin(), g.keys.end(), next);
        if (it != g.keys.end() && *it == next)
          result.push_back(static_cast<Row>(it - g.keys.begin()));
      }
    std::sort(result.begin(), result.end());
    return result;
  }

  Row find_row(const Groups<String>& g, Row gid,
               const std::vector<String>& secondary, String key) const {
    auto begin = g.rows.begin() + g.ptr[gid],
         end = g.rows.begin() + g.ptr[gid + 1];
    auto it = std::lower_bound(begin, end, key, [&](Row r, String value) {
      return secondary[r] < value;
    });
    return it != end && secondary[*it] == key ? *it : -1;
  }

  // Intersect observed string IDs with one actual determinant group. Complete
  // groups use direct addressing into their existing row list, without padding.
  template <class F>
  void join_group(const Groups<String>& g, Row gid, std::size_t all_secondary,
                  const Row* begin, const Row* end, int degree, F fn) const {
    if (begin == end) return;
    std::size_t lo = g.ptr[gid], hi = g.ptr[gid + 1];
    if (hi - lo == all_secondary) {
      for (auto p = begin; p != end; ++p) fn(g.rows[lo + *p], degree);
    } else if (hi - lo > 8 * static_cast<std::size_t>(end - begin)) {
      auto first = g.secondary_id.begin() + lo,
           last = g.secondary_id.begin() + hi;
      for (auto p = begin; p != end; ++p) {
        first = std::lower_bound(first, last, *p);
        if (first == last) break;
        if (*first == *p) fn(g.rows[first - g.secondary_id.begin()], degree);
      }
    } else {
      auto p = begin;
      while (lo < hi && p != end) {
        Row b = g.secondary_id[lo];
        if (b < *p)
          ++lo;
        else if (*p < b)
          ++p;
        else {
          fn(g.rows[lo], degree);
          ++lo;
          ++p;
        }
      }
    }
  }

  template <class F>
  void same_spin(Row i, bool change_alpha, F fn) const {
    const Groups<String>& group = change_alpha ? gb : ga;
    const auto& changing = change_alpha ? alpha : beta;
    const Row gid = group.id[i];
    const String x = changing[i];
    const Groups<String>& strings = change_alpha ? ga : gb;
    const Singles& single = change_alpha ? sa : sb;
    const Singles& doub = change_alpha ? da : db;
    Row string_id = strings.id[i];
    std::size_t group_size = group.ptr[gid + 1] - group.ptr[gid];
    if (single.cached && doub.cached) {
      std::size_t links = single.ptr[string_id + 1] - single.ptr[string_id] +
                          doub.ptr[string_id + 1] - doub.ptr[string_id];
      if (group_size == strings.keys.size() || group_size > links + 1) {
        static const Row empty = 0;
        auto emit = [&](const Singles& graph, int degree) {
          const Row* base =
              graph.targets.empty() ? &empty : graph.targets.data();
          join_group(group, gid, strings.keys.size(),
                     base + graph.ptr[string_id],
                     base + graph.ptr[string_id + 1], degree, fn);
        };
        emit(single, 1);
        emit(doub, 2);
        return;
      }
    }
    const int occ = change_alpha ? noa : nob, vir = norb - occ;
    const std::size_t physical =
        occ * vir +
        static_cast<std::size_t>(occ * (occ - 1) / 2) * (vir * (vir - 1) / 2);
    if (group.ptr[gid + 1] - group.ptr[gid] <= physical + 1) {
      for (std::size_t p = group.ptr[gid]; p < group.ptr[gid + 1]; ++p) {
        Row j = group.rows[p];
        int bits = popcount(x ^ changing[j]);
        if (bits == 2 || bits == 4) fn(j, bits / 2);
      }
    } else {
      std::vector<int> os, vs;
      for (int p = 0; p < norb; ++p) ((x >> p) & 1 ? os : vs).push_back(p);
      auto emit = [&](String value, int degree) {
        Row j = find_row(group, gid, changing, value);
        if (j >= 0) fn(j, degree);
      };
      for (int h : os)
        for (int p : vs) emit(x ^ (String{1} << h) ^ (String{1} << p), 1);
      for (std::size_t h = 0; h < os.size(); ++h)
        for (std::size_t k = 0; k < h; ++k)
          for (std::size_t p = 0; p < vs.size(); ++p)
            for (std::size_t q = 0; q < p; ++q)
              emit(x ^ (String{1} << os[h]) ^ (String{1} << os[k]) ^
                       (String{1} << vs[p]) ^ (String{1} << vs[q]),
                   2);
    }
  }

  struct NeighborRange {
    const Row* begin;
    const Row* end;
  };

  NeighborRange single_neighbors(Row i, const Groups<String>& groups,
                                 const Singles& graph,
                                 std::vector<Row>& scratch) const {
    static const Row empty = 0;
    const Row id = groups.id[i];
    if (graph.cached) {
      const Row* base = graph.targets.empty() ? &empty : graph.targets.data();
      return {base + graph.ptr[id], base + graph.ptr[id + 1]};
    }
    scratch = generated_singles(groups.keys[id], groups);
    const Row* base = scratch.empty() ? &empty : scratch.data();
    return {base, base + scratch.size()};
  }

  template <class F>
  static void emit_bitmap_matches(const GroupBitmap& bitmap,
                                  const NeighborWords& neighbors, Row group_id,
                                  const Row* group_rows, std::size_t word_begin,
                                  std::size_t word_end, F& emit) {
    const auto base = static_cast<std::size_t>(group_id) * bitmap.stride;
    for (auto p = word_begin; p < word_end; ++p) {
      const auto w = static_cast<std::size_t>(neighbors.word[p]);
      const unsigned long present = bitmap.bits[base + w];
      for (unsigned long hits = present & neighbors.mask[p]; hits; hits &= hits - 1) {
        const unsigned long bit = hits & -hits;
        const Row offset = bitmap.rank[base + w] + __builtin_popcountl(present & (bit - 1));
        emit(group_rows[offset], 2);
      }
    }
  }

  template <class F>
  void opposite_spin(Row i, F emit, std::uint8_t* marks) const {
    std::vector<Row> alpha_scratch, beta_scratch;
    const auto alpha_neighbors = single_neighbors(i, ga, sa, alpha_scratch);
    const auto beta_neighbors = single_neighbors(i, gb, sb, beta_scratch);
    if (alpha_neighbors.begin == alpha_neighbors.end ||
        beta_neighbors.begin == beta_neighbors.end) return;

    const bool beta_outer = sa.cached && sb.cached &&
                            beta_scan_work[gb.id[i]] < alpha_scan_work[ga.id[i]];
    const auto& outer = beta_outer ? gb : ga;
    const auto& inner = beta_outer ? ga : gb;
    const auto& inner_strings = beta_outer ? alpha : beta;
    const auto& outer_neighbors = beta_outer ? beta_neighbors : alpha_neighbors;
    const auto& inner_neighbors = beta_outer ? alpha_neighbors : beta_neighbors;
    const auto inner_count = inner_neighbors.end - inner_neighbors.begin;
    bool marked = false;

    const auto& bitmap = beta_outer ? bgb : bga;
    const auto& bit_neighbors = beta_outer ? bsa : bsb;
    const Row inner_id = inner.id[i];
    const bool has_bitmap = !bitmap.bits.empty();
    const std::size_t word_begin = has_bitmap ? bit_neighbors.ptr[inner_id] : 0;
    const std::size_t word_end = has_bitmap ? bit_neighbors.ptr[inner_id + 1] : 0;

    for (auto p = outer_neighbors.begin; p != outer_neighbors.end; ++p) {
      const Row group_id = *p;
      const auto lo = outer.ptr[group_id], hi = outer.ptr[group_id + 1];
      const auto size = hi - lo;
      if (size == inner.keys.size()) {
        for (auto b = inner_neighbors.begin; b != inner_neighbors.end; ++b)
          emit(outer.rows[lo + *b], 2);
      } else if (has_bitmap && size > 2 * (word_end - word_begin)) {
        emit_bitmap_matches(bitmap, bit_neighbors, group_id,
                            outer.rows.data() + lo, word_begin, word_end, emit);
      } else if (size <= inner_count * outer.scan_weight[group_id]) {
        if (marks) {
          // Mark once per input row, then reuse across scanned groups.
          if (!marked) {
            for (auto b = inner_neighbors.begin; b != inner_neighbors.end; ++b)
              marks[*b] = 1;
            marked = true;
          }
          for (auto q = lo; q < hi; ++q)
            if (marks[outer.secondary_id[q]]) emit(outer.rows[q], 2);
        } else {
          for (auto q = lo; q < hi; ++q) {
            const Row j = outer.rows[q];
            if (popcount(inner_strings[i] ^ inner_strings[j]) == 2) emit(j, 2);
          }
        }
      } else {
        join_group(outer, group_id, inner.keys.size(), inner_neighbors.begin,
                   inner_neighbors.end, 2, emit);
      }
    }
    if (marked)
      for (auto b = inner_neighbors.begin; b != inner_neighbors.end; ++b)
        marks[*b] = 0;
  }

  template <class F>
  void neighbors(Row i, F fn, std::uint8_t* marks = nullptr) const {
    same_spin(i, true, fn);
    same_spin(i, false, fn);
    opposite_spin(i, fn, marks);
  }

  double value(Row i, Row j, int degree) const {
    const unsigned long* bra = onv + static_cast<std::size_t>(i) * width;
    const unsigned long* ket = onv + static_cast<std::size_t>(j) * width;
    if (degree == 1) {
      if (width == 1) return squant::get_HijS_cpu(bra, ket, h1, h2, sorb, 1);
      return squant::get_HijS_cpu(bra, ket, h1, h2, sorb, width);
    }
    int p[2], q[2];
    if (width == 1)
      squant::diff_orb_cpu(bra, ket, 1, p, q);
    else
      squant::diff_orb_cpu(bra, ket, width, p, q);
    const unsigned long* pi =
        fermion_parity.data() + static_cast<std::size_t>(i) * width;
    const unsigned long* pj =
        fermion_parity.data() + static_cast<std::size_t>(j) * width;
    int odd;
    if (width == 1)
      odd = ((pi[0] >> p[0]) ^ (pi[0] >> p[1]) ^ (pj[0] >> q[0]) ^
             (pj[0] >> q[1])) &
            1;
    else
      odd = ((pi[p[0] / 64] >> (p[0] % 64)) ^ (pi[p[1] / 64] >> (p[1] % 64)) ^
             (pj[q[0] / 64] >> (q[0] % 64)) ^ (pj[q[1] / 64] >> (q[1] % 64))) &
            1;
    int sgn = 1 - 2 * odd;
    // Numerical body copied from PyNQS get_HijD_cpu; only orbital parsing
    // and its four parity evaluations have been moved/cached above.
    double Hij = squant::h2e_get_cpu(h2, p[0], p[1], q[0], q[1]);
    Hij *= static_cast<double>(sgn);
    return Hij;
  }

  double reference_value(Row i, Row j, int degree) const {
    const unsigned long* bra = onv + static_cast<std::size_t>(i) * width;
    const unsigned long* ket = onv + static_cast<std::size_t>(j) * width;
    if (degree == 1) return squant::get_HijS_cpu(bra, ket, h1, h2, sorb, width);
    return squant::get_HijD_cpu(bra, ket, h1, h2, sorb, width);
  }

 public:
  FixedSpaceHamiltonian(py::array_t<std::uint8_t, py::array::c_style> d,
                        py::array_t<double, py::array::c_style> h1e,
                        py::array_t<double, py::array::c_style> h2e, int s,
                        int ne, int a, int b, std::size_t link_budget)
      : onv_owner(d),
        h1_owner(h1e),
        h2_owner(h2e),
        sorb(s),
        norb(s / 2),
        nele(ne),
        noa(a),
        nob(b) {
    if (s < 2 || s % 2 || s > 16 * sizeof(String) || ne > MAX_NELE)
      throw std::invalid_argument(
          "strings requires even sorb <= 256 and matching electron capacity");
    if (d.ndim() != 2 || d.shape(0) < 1 ||
        d.shape(0) > std::numeric_limits<Row>::max() ||
        d.shape(1) != 8 * ((s + 63) / 64))
      throw std::invalid_argument("invalid packed determinant shape");
    n = d.shape(0);
    width = d.shape(1) / 8;
    onv = reinterpret_cast<const unsigned long*>(d.data());
    h1 = h1e.data();
    h2 = h2e.data();
    std::size_t pair = static_cast<std::size_t>(s) * (s - 1) / 2;
    if (h1e.size() != s * s || h2e.size() != pair * (pair + 1) / 2)
      throw std::invalid_argument("invalid compressed integral sizes");
    alpha.resize(n);
    beta.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      for (int w = 0; w < width; ++w) {
        alpha[i] |= static_cast<String>(compact_even(onv[i * width + w]))
                    << (32 * w);
        beta[i] |= static_cast<String>(compact_even(onv[i * width + w] >> 1))
                   << (32 * w);
      }
    }
    ga = make_groups(alpha, beta);
    gb = make_groups(beta, alpha);
    ga.secondary_id.resize(n);
    gb.secondary_id.resize(n);
    for (std::size_t p = 0; p < n; ++p) {
      ga.secondary_id[p] = gb.id[ga.rows[p]];
      gb.secondary_id[p] = ga.id[gb.rows[p]];
    }
    sa = make_singles(ga.keys, norb, noa, link_budget / 2);
    sb = make_singles(gb.keys, norb, nob, link_budget / 2);
    alpha_scan_work = scan_work(sa, ga);
    beta_scan_work = scan_work(sb, gb);
    // Sparse groups are cheaper to scan; amortize double-link setup only
    // when a spin string occurs repeatedly with different spectators.
    if (n > ga.keys.size() * 4 && link_bytes(sa) < link_budget / 2)
      da = make_doubles(ga.keys, norb, noa, link_budget / 2 - link_bytes(sa));
    if (n > gb.keys.size() * 4 && link_bytes(sb) < link_budget / 2)
      db = make_doubles(gb.keys, norb, nob, link_budget / 2 - link_bytes(sb));

    const double bitmap_bound =
        (ga.keys.size() * double((gb.keys.size() + 63) / 64) +
         gb.keys.size() * double((ga.keys.size() + 63) / 64)) * 12 +
        (sa.targets.size() + sb.targets.size()) * 12.0 +
        (ga.keys.size() + gb.keys.size() + 2) * 8.0;
    const auto cached_bytes = link_bytes(sa) + link_bytes(sb) +
                              link_bytes(da) + link_bytes(db);
    if (sa.cached && sb.cached && n > ga.keys.size() * 32 &&
        n > gb.keys.size() * 32 && cached_bytes + bitmap_bound +
        std::max(ga.keys.size(), gb.keys.size()) * double(omp_get_max_threads()) < link_budget) {
      bga = group_bitmap(ga, gb.keys.size());
      bgb = group_bitmap(gb, ga.keys.size());
      bsa = neighbor_words(sa);
      bsb = neighbor_words(sb);
    }

    const std::size_t keys = std::max(ga.keys.size(), gb.keys.size());
    if (sa.cached && sb.cached &&
        keys * sizeof(std::uint8_t) * omp_get_max_threads() <= link_budget)
      mixed_marker_size = keys;
    diag.resize(n);
    fermion_parity.assign(n * width, 0);
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
      diag[i] = squant::get_Hii_cpu(onv + i * width, onv + i * width, h1, h2,
                                    sorb, nele, width);
      for (int p = 0; p < sorb; ++p)
        if (squant::parity_cpu(onv + i * width, p) < 0)
          fermion_parity[i * width + p / 64] |= 1ULL << (p % 64);
    }
  }

  py::array_t<double> diagonal() const {
    py::array_t<double> out(n);
    std::copy(diag.begin(), diag.end(), out.mutable_data());
    return out;
  }

  py::array_t<double> matvec(py::array_t<double, py::array::c_style> v) const {
    if (v.ndim() != 1 || v.size() != n)
      throw std::invalid_argument("incorrect vector shape");
    py::array_t<double> out(n);
    const double* x = v.data();
    double* y = out.mutable_data();
    py::gil_scoped_release release;
#pragma omp parallel
    {
      std::vector<std::uint8_t> marks(mixed_marker_size, 0);
#pragma omp for
      for (std::size_t row = 0; row < n; ++row) {
        Row i = static_cast<Row>(row);
        double sum = diag[i] * x[i];
        neighbors(
            i, [&](Row j, int degree) { sum += value(i, j, degree) * x[j]; },
            marks.empty() ? nullptr : marks.data());
        y[i] = sum;
      }
    }
    return out;
  }

  // Diagnostic only: retain one row's connections, then time lookup, SC and
  // accumulation separately. Normal matvec has no counters or clock reads.
  py::dict profile(py::array_t<double, py::array::c_style> v,
                   std::size_t requested_rows, int repeats) const {
    if (v.ndim() != 1 || v.size() != n || requested_rows < 1 || repeats < 1)
      throw std::invalid_argument("invalid profile vector/rows/repeats");
    using Clock = std::chrono::steady_clock;
    auto seconds = [](Clock::time_point start) {
      return std::chrono::duration<double>(Clock::now() - start).count();
    };
    const std::size_t count = std::min(n, requested_rows);
    std::vector<Row> row_ids(count);
    std::vector<double> sigma(count);
    std::vector<Row> targets[3];
    std::vector<double> values[3];
    std::uint64_t links[3] = {0, 0, 0};
    double lookup[3] = {0, 0, 0}, sc[3] = {0, 0, 0},
           reference_sc[3] = {0, 0, 0};
    double accumulation = 0, max_sc_error = 0;
    const double* x = v.data();
    std::vector<std::uint8_t> marks(mixed_marker_size, 0);
    const auto wall = Clock::now();
    {
      py::gil_scoped_release release;
      for (std::size_t k = 0; k < count; ++k) {
        Row i = static_cast<Row>(k * n / count);
        row_ids[k] = i;
        for (auto& list : targets) list.clear();
        auto emit_same = [&](Row j, int degree) {
          targets[degree - 1].push_back(j);
        };
        auto start = Clock::now();
        same_spin(i, true, emit_same);
        lookup[0] += seconds(start);
        start = Clock::now();
        same_spin(i, false, emit_same);
        lookup[1] += seconds(start);
        start = Clock::now();
        opposite_spin(
            i, [&](Row j, int) { targets[2].push_back(j); },
            marks.empty() ? nullptr : marks.data());
        lookup[2] += seconds(start);
        for (int kind = 0; kind < 3; ++kind) {
          links[kind] += targets[kind].size();
          values[kind].resize(targets[kind].size());
          start = Clock::now();
          for (int r = 0; r < repeats; ++r)
            for (std::size_t p = 0; p < targets[kind].size(); ++p)
              values[kind][p] = value(i, targets[kind][p], kind == 0 ? 1 : 2);
          sc[kind] += seconds(start) / repeats;
          start = Clock::now();
          for (std::size_t p = 0; p < targets[kind].size(); ++p)
            max_sc_error = std::max(
                max_sc_error,
                std::abs(values[kind][p] - reference_value(i, targets[kind][p],
                                                           kind == 0 ? 1 : 2)));
          reference_sc[kind] += seconds(start);
        }
        start = Clock::now();
        double sum = 0;
        for (int r = 0; r < repeats; ++r) {
          sum = diag[i] * x[i];
          for (int kind = 0; kind < 3; ++kind)
            for (std::size_t p = 0; p < targets[kind].size(); ++p)
              sum += values[kind][p] * x[targets[kind][p]];
        }
        accumulation += seconds(start) / repeats;
        sigma[k] = sum;
      }
    }
    py::dict times, counts, out;
    times["alpha_lookup_s"] = lookup[0];
    times["beta_lookup_s"] = lookup[1];
    times["mixed_lookup_s"] = lookup[2];
    times["single_sc_s"] = sc[0];
    times["same_double_sc_s"] = sc[1];
    times["mixed_double_sc_s"] = sc[2];
    times["accumulate_s"] = accumulation;
    counts["singles"] = links[0];
    counts["same_spin_doubles"] = links[1];
    counts["opposite_spin_doubles"] = links[2];
    out["sampled_rows"] = count;
    out["sc_repeats"] = repeats;
    out["wall_s"] = seconds(wall);
    out["phase_s"] = times;
    out["connections"] = counts;
    out["original_slater_condon_s"] =
        py::make_tuple(reference_sc[0], reference_sc[1], reference_sc[2]);
    out["slater_condon_max_abs_error"] = max_sc_error;
    out["rows"] = row_ids;
    out["sigma"] = sigma;
    return out;
  }

  py::dict info() const {
    py::dict out;
    out["n_alpha_strings"] = ga.keys.size();
    out["n_beta_strings"] = gb.keys.size();
    out["alpha_single_links"] = sa.targets.size();
    out["beta_single_links"] = sb.targets.size();
    out["alpha_links_cached"] = sa.cached;
    out["beta_links_cached"] = sb.cached;
    out["alpha_double_links_cached"] = da.cached;
    out["beta_double_links_cached"] = db.cached;
    out["alpha_double_links"] = da.targets.size();
    out["beta_double_links"] = db.targets.size();
    out["parallel_runtime"] = "openmp";
    out["threads"] = omp_get_max_threads();
    out["openmp_dynamic"] = bool(omp_get_dynamic());
    out["openmp_schedule"] = "default";
    out["parity_bytes"] = fermion_parity.size() * sizeof(unsigned long);
    out["onv_words"] = width;
    out["spin_string_bits"] = sizeof(String) * 8;
    out["mixed_lookup_workspace_bytes"] =
        mixed_marker_size * sizeof(std::uint8_t) * omp_get_max_threads();
    out["offdiagonal_value_cache_bytes"] = 0;

    const auto bitmap_bytes = (bga.bits.size() + bgb.bits.size() +
                               bsa.mask.size() + bsb.mask.size()) * 8 +
                              (bga.rank.size() + bgb.rank.size() +
                               bsa.word.size() + bsb.word.size()) * 4 +
                              (bsa.ptr.size() + bsb.ptr.size()) * 8;
    out["mixed_bitmap_bytes"] = bitmap_bytes;
    out["connectivity_bytes"] = bitmap_bytes +
        (alpha.size() + beta.size() + ga.keys.size() + gb.keys.size()) *
            sizeof(String) +
        (ga.ptr.size() + gb.ptr.size() + sa.ptr.size() + sb.ptr.size() +
         da.ptr.size() + db.ptr.size() + ga.scan_weight.size() +
         gb.scan_weight.size() + alpha_scan_work.size() +
         beta_scan_work.size()) *
            8 +
        (ga.rows.size() + gb.rows.size() + ga.id.size() + gb.id.size() +
         ga.secondary_id.size() + gb.secondary_id.size() + sa.targets.size() +
         sb.targets.size() + da.targets.size() + db.targets.size()) *
            4;
    return out;
  }
};
}  // namespace

template <class String>
static void bind_string(py::module_& m, const char* name) {
  using Hamiltonian = FixedSpaceHamiltonian<String>;
  py::class_<Hamiltonian>(m, name)
      .def("diagonal", &Hamiltonian::diagonal)
      .def("matvec", &Hamiltonian::matvec)
      .def("profile", &Hamiltonian::profile, py::arg("v"),
           py::arg("rows") = 512, py::arg("repeats") = 3)
      .def("info", &Hamiltonian::info);
}

void bind_sci(py::module_& m) {
  bind_string<unsigned long>(m, "_StringHamiltonian64");
  bind_string<unsigned __int128>(m, "_StringHamiltonian128");
  // One algorithm; choose only the integer width used for spin-string keys.
  m.def("StringHamiltonian",
        [](py::array_t<std::uint8_t, py::array::c_style> onv,
           py::array_t<double, py::array::c_style> h1,
           py::array_t<double, py::array::c_style> h2, int sorb, int nele,
           int noa, int nob, std::size_t link_budget) -> py::object {
          if (sorb <= 128)
            return py::cast(FixedSpaceHamiltonian<unsigned long>(
                onv, h1, h2, sorb, nele, noa, nob, link_budget));
          return py::cast(FixedSpaceHamiltonian<unsigned __int128>(
              onv, h1, h2, sorb, nele, noa, nob, link_budget));
        });
}
