#include "osd/osd_types.h"

template <typename T>
void do_test() {
  std::list<T*> l;
  T::generate_test_instances(l);
}

// generate_test_instances
void test_1() {
  // osd_reqid_t
  do_test<osd_reqid_t>();
  // object_locator_t
  do_test<object_locator_t>();
  // request_redirect_t
  do_test<request_redirect_t>();
  // objectstore_perf_stat_t
  do_test<objectstore_perf_stat_t>();
  // osd_stat_t
  do_test<osd_stat_t>();
  // pg_t
  do_test<pg_t>();
  // coll_t
  do_test<coll_t>();
  // pool_snap_info_t
  do_test<pool_snap_info_t>();
  // pg_pool_t
  do_test<pg_pool_t>();
  // object_stat_sum_t
  do_test<object_stat_sum_t>();
  // object_stat_collection_t
  do_test<object_stat_collection_t>();
  // pg_stat_t
  do_test<pg_stat_t>();
  // pool_stat_t
  do_test<pool_stat_t>();
  // pg_history_t
  do_test<pg_history_t>();
  // pg_info_t
  do_test<pg_info_t>();
  // pg_notify_t */
  do_test<pg_notify_t>();
  // pg_interval_t
  do_test<PastIntervals::pg_interval_t>();
  // PastIntervals
  do_test<PastIntervals>();
  // pg_query_t
  do_test<pg_query_t>();
  // ObjectModDesc
  do_test<ObjectModDesc>();
  // pg_log_entry_t
  do_test<pg_log_entry_t>();
  // pg_log_dup_t
  do_test<pg_log_dup_t>();
  // pg_log_t
  do_test<pg_log_t>();
  // object_copy_cursor_t
  do_test<object_copy_cursor_t>();
  // object_copy_data_t
  do_test<object_copy_data_t>();
  // pg_create_t
  do_test<pg_create_t>();
  // pg_hit_set_info_t
  do_test<pg_hit_set_info_t>();
  // pg_hit_set_history_t
  do_test<pg_hit_set_history_t>();
  // OSDSuperblock
  do_test<OSDSuperblock>();
  // SnapSet
  do_test<SnapSet>();
  // watch_info_t
  do_test<watch_info_t>();
  // object_manifest_t
  do_test<object_manifest_t>();
  // object_info_t - FIXME
  do_test<object_info_t>();
  // ObjectRecoveryProgress
  do_test<ObjectRecoveryProgress>();
  // ObjectRecoveryInfo
  do_test<ObjectRecoveryInfo>();
  // PushReplyOp
  do_test<PushReplyOp>();
  // PullOp
  do_test<PullOp>();
  // ScrubMap
  do_test<ScrubMap>();
  // ScrubMap::object
  do_test<ScrubMap::object>();
}

// Test osd_reqid_t
void test_2() {

  {
    osd_reqid_t reqid(entity_name_t::CLIENT(1), utime_t(1, 2), 0);
  }

  {
    osd_reqid_t reqid;
    assert(reqid.tid == 0);
    assert(reqid.inc == 0);
  }

  {
    osd_reqid_t reqid;
    reqid.name = entity_name_t::CLIENT(1);
    reqid.tid = utime_t(2, 3);
    reqid.inc = 0;
  }
  
}

// Test object_locator_t
void test_3() {
  {
    object_locator_t oloc;
    assert(oloc.pool == -1);
    assert(oloc.hash == -1);
  }
  {
    object_locator_t oloc(123);
    assert(oloc.pool == 123);
  }
  {
    object_locator_t oloc(123, 456);
    assert(oloc.pool == 123);
    assert(oloc.hash == 456);
  }
  {
    object_locator_t oloc(123, "abc");
    assert(oloc.pool == 123);
    assert(oloc.nspace == "abc");
  }
  {
    object_locator_t oloc(123, "abc", 456);
    assert(oloc.pool == 123);
    assert(oloc.nspace == "abc");
    assert(oloc.hash == 456);
  }
  {
    object_locator_t oloc(123, "abc", "xyz");
    assert(oloc.pool == 123);
    assert(oloc.nspace == "abc");
    assert(oloc.key == "xyz");
    assert(oloc.hash == -1);
    assert(oloc.get_pool() == 123);
    oloc.clear();
    assert(oloc.pool == -1);
    assert(oloc.key == "");
    assert(oloc.nspace == "");
    assert(oloc.hash == -1);
    assert(oloc.empty());
  }
  {
    const hobject_t soid;
    object_locator_t oloc(soid);
    assert(oloc.hash == -1);
  }
}

// Test request_redirect_t
void test_4() {
  request_redirect_t(object_locator_t(), 123);
  request_redirect_t(object_locator_t());
  request_redirect_t rdd;
  assert(rdd.empty());
}

// Test OSD op flags
void test_5() {
  std::cout << CEPH_OSD_RMW_FLAG_READ << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_WRITE << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_CLASS_READ << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_CLASS_WRITE << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_PGOP << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_CACHE << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_FORCE_PROMOTE << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_SKIP_HANDLE_CACHE << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_SKIP_PROMOTE << '\n';
  std::cout << CEPH_OSD_RMW_FLAG_RWORDERED << '\n';
}

// Test pg_t
void test_6() {
  {
    pg_t pg;
    assert(pg.m_pool == 0);
    assert(pg.m_seed == 0);
    assert(pg.m_preferred == -1);
  }
  {
    pg_t pg(123, 456);
    assert(pg.m_pool == 456);
    assert(pg.m_seed == 123);
    assert(pg.m_preferred == -1);
  }
  {
    old_pg_t opg;
    pg_t pg(opg.v);
  }
  {
    pg_t pg;
    pg.set_ps(123);
    assert(pg.m_seed == 123);
    pg.set_pool(456);
    assert(pg.m_pool == 456);
    pg.set_preferred(1);
    assert(pg.m_preferred == 1);
  }
}

// Test spg_t
void test_7() {
  {
    spg_t spg;
    assert(spg.shard == shard_id_t::NO_SHARD);
  }
  {
    pg_t pg(123, 456);
    spg_t spg(pg, shard_id_t::NO_SHARD);
    assert(spg.pgid.m_pool == 456);
    assert(spg.pgid.m_seed == 123);
    assert(spg.pgid.m_preferred == -1);
  }
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5(); test_6(); test_7();
  return 0;
}
