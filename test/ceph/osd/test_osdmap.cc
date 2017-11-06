#include "osd/OSDMap.h"

#include <cassert>

void test_1() {

  std::list<osd_info_t*> o1;
  osd_info_t::generate_test_instances(o1);
}

void test_2() {

  { // osd_info_t
    osd_info_t oi;
    assert(oi.last_clean_begin == 0);
    assert(oi.last_clean_end == 0);
    assert(oi.up_from == 0);
    assert(oi.up_thru == 0);
    assert(oi.down_at == 0);
    assert(oi.lost_at == 0);

    XMLFormatter f;
    oi.dump(&f);
  }

  { // osd_xinfo_t
    osd_xinfo_t ox;
    assert(ox.laggy_probability == 0);
    assert(ox.laggy_interval == 0);
    assert(ox.features == 0);
    assert(ox.old_weight == 0);
  }

  { // OSDMap
    OSDMap map;
    assert(map.epoch == 0);
    assert(map.pool_max == 0);
    assert(map.flags == 0);
    assert(map.num_osd == 0);
    assert(map.num_up_osd == 0);
    assert(map.num_in_osd == 0);
  }
}

void test_3() {
  OSDMap map;
  const OSDMap old;
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
