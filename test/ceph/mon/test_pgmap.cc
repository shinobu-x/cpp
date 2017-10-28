#include "mon/PGMap.h"
#include "include/stringify.h"

class test_texttable : public TextTable {
public:
  test_texttable(bool verbose) {
    for (int i = 0; i < 4; ++i)
      define_column("", TextTable::LEFT, TextTable::LEFT);

    if (verbose)
      for (int i = 0; i < 4; ++i)
        define_column("", TextTable::LEFT, TextTable::LEFT);
  }

  const std::string& get(unsigned r, unsigned c) const {
    assert(r < row.size());
    assert(c < row[r].size());
    return row[r][c];
  }

  static std::string percentify(float a);
};

std::string percentify(float a) {
  std::stringstream ss;
  if (a < 0.01)
    ss << "0";
  else
    ss << std::fixed << std::setprecision(2) << a;
  return ss.str();
}

void test_1() {
  { // dump_object_stat_sum
    bool verbose = true;
    test_texttable tbl(verbose);
    object_stat_sum_t sum;
    sum.num_bytes = 42*1024*1024;
    sum.num_objects = 42;
    sum.num_objects_degraded = 13;
    sum.num_objects_dirty = 3;
    sum.num_rd = 100;
    sum.num_rd_kb = 123;
    sum.num_wr = 101;
    sum.num_wr_kb = 321;

    sum.calc_copies(3);
    uint64_t avail = 2017*1024*1024;
    pg_pool_t pool;
    pool.quota_max_objects = 2000;
    pool.quota_max_bytes = 2000*1024*1024;
    pool.type = pg_pool_t::TYPE_REPLICATED;
    PGMap::dump_object_stat_sum(
      tbl, nullptr, sum, avail, pool.get_size(), verbose, &pool);
    std::cout << stringify(si_t(sum.num_bytes)) << '\n';
    std::cout << tbl.get(0, 0) << '\n';
    assert(stringify(si_t(sum.num_bytes)) == tbl.get(0, 0));

    float copies_rate =
      (static_cast<float>(sum.num_object_copies - sum.num_objects_degraded) /
        sum.num_object_copies);
    float used_bytes = sum.num_bytes * copies_rate;
    float used_percent = used_bytes / (used_bytes + avail) * 100;
    unsigned col = 0;

    assert(stringify(si_t(sum.num_bytes)) == tbl.get(0, col++));
    assert(percentify(used_percent) == tbl.get(0, col++));
    assert(stringify(si_t(avail / pool.size)) == tbl.get(0, col++));
    assert(stringify(sum.num_objects) == tbl.get(0, col++));
    assert(stringify(si_t(sum.num_objects_dirty)) == tbl.get(0, col++));
    assert(stringify(si_t(sum.num_rd)) == tbl.get(0, col++));
    assert(stringify(si_t(sum.num_wr)) == tbl.get(0, col++));

    uint64_t raw_bytes_used = sum.num_bytes * pool.get_size() * copies_rate;
    assert(stringify(si_t(raw_bytes_used)) == tbl.get(0, col++));
  }
}
auto main() -> decltype(0) {
  test_1();
  return 0;
}
