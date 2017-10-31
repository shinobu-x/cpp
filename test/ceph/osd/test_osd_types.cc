#include "osd/osd_types.h"

template <typename T>
void do_test() {
  std::list<T*> l;
  T::generate_test_instances(l);
}

void test_1() {
/*
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

//  do_test<PastIntervals::pg_interval_t>();
  // compact_interval_t
//  do_test<compact_interval_t>();
  // pi_compact_rep
//  do_test<pi_compact_rep>();
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
