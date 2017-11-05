#include "osd/OSDMap.h"

#include <cassert>

void test_1() {

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

void test_2() {
  OSDMap map;
  const OSDMap old;

  int r = map.Incremental::get_net_marked_out(&map);
  (void)r;

  r = map.Incremental::get_net_marked_down(&map);
  (void)r;

  uuid_d uuid;
  r = map.Incremental::identify_osd(uuid);
  (void)r;
  (void)uuid;

  r = map.Incremental::propagate_snap_to_tiers(&cct, old);
  (void)r;
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
