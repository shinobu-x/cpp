#include "include/types.h"
#include "osd/osd_types.h"
#include "osd/OSDMap.h"
#include "gtest/gtest.h"
#include "common/Thread.h"
#include "include/stringify.h"
#include "osd/ReplicatedBackend.h"

#include <iostream>

void test_1() {
  uint32_t mask = 0xE947FA20;
  uint32_t bits = 12;
  int64_t pool = 0;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000000.02A"));
  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));
  std::cout << prefixes_correct << std::endl;
  std::cout << prefixes_out << std::endl;
}

void test_2() {
  uint32_t mask = 0x0000000F;
  uint32_t bits = 6;
  int64_t pool = 20;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000014.F0"));
  prefixes_correct.insert(std::string("0000000000000014.F4"));
  prefixes_correct.insert(std::string("0000000000000014.F8"));
  prefixes_correct.insert(std::string("0000000000000014.FC"));

  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));
  std::cout << prefixes_correct << std::endl;
  std::cout << prefixes_out << std::endl;
}

void test_3() {
  uint32_t mask = 0xDEADBEAF;
  uint32_t bits = 25;
  int64_t pool = 0;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000000.FAEBDA0"));
  prefixes_correct.insert(std::string("0000000000000000.FAEBDA2"));
  prefixes_correct.insert(std::string("0000000000000000.FAEBDA4"));
  prefixes_correct.insert(std::string("0000000000000000.FAEBDA6"));
  prefixes_correct.insert(std::string("0000000000000000.FAEBDA8"));
  prefixes_correct.insert(std::string("0000000000000000.FAEBDAA"));
  prefixes_correct.insert(std::string("0000000000000000.FAEBDAC"));
  prefixes_correct.insert(std::string("0000000000000000.FAEBDAE"));

  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));

  std::cout << prefixes_correct << std::endl;
  std::cout << prefixes_out << std::endl;
}

void test_4() {
  uint32_t mask = 0xE947FA20;
  uint32_t bits = 32;
  int64_t pool = 0x23;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000023.02AF749E"));
  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));

  std::cout << prefixes_correct << std::endl;
  std::cout << prefixes_out << std::endl;
}

void test_5() {
  uint32_t mask = 0xE947FA20;
  uint32_t bits = 0;
  int64_t pool = 0x23;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000023."));

  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));

  std::cout << prefixes_correct << std::endl;
  std::cout << prefixes_out << std::endl;
}

void test_6() {
  uint32_t mask = 0xDEADBEAF;
  uint32_t bits = 1;
  int64_t pool = 0x34AC5D00;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000034AC5D00.1"));
  prefixes_correct.insert(std::string("0000000034AC5D00.3"));
  prefixes_correct.insert(std::string("0000000034AC5D00.5"));
  prefixes_correct.insert(std::string("0000000034AC5D00.7"));
  prefixes_correct.insert(std::string("0000000034AC5D00.9"));
  prefixes_correct.insert(std::string("0000000034AC5D00.B"));
  prefixes_correct.insert(std::string("0000000034AC5D00.D"));
  prefixes_correct.insert(std::string("0000000034AC5D00.F"));

  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));

  std::cout << prefixes_correct << std::endl;
  std::cout << prefixes_out << std::endl;
}

void test_7() {
  for (unsigned i = 0; i < 4; ++i) {
    int osd = 1;
    epoch_t e = 40;
    std::shared_ptr<OSDMap> osdmap(new OSDMap());
    osdmap->set_max_osd(10);
    osdmap->set_state(osd, CEPH_OSD_EXISTS);
    osdmap->set_epoch(e);

    std::shared_ptr<OSDMap> last_osdmap(new OSDMap());
    last_osdmap->set_max_osd(10);
    last_osdmap->set_state(osd, CEPH_OSD_EXISTS);
    last_osdmap->set_epoch(e);

    epoch_t same_interval_since = e;
    epoch_t last_epoch_clean = same_interval_since;

    int64_t pool = 200;
    int pg_num = 4;
    __u8 min_size = 2;

    boost::scoped_ptr<IsPGRecoverablePredicate> recoverable(
      new ReplicatedBackend::RPCRecPred());

    {
      OSDMap::Incremental inc(e + 1);
      inc.new_pools[pool].min_size = min_size;
      inc.new_pools[pool].set_pg_num(pg_num);
      inc.new_up_thru[osd] = e + 1;
      osdmap->apply_incremental(inc);
      last_osdmap->apply_incremental(inc);
    }
    std::vector<int> new_acting;
    new_acting.push_back(osd);
    new_acting.push_back(osd + 1);
    std::vector<int> old_acting = new_acting;
    int new_primary = osd;
    int old_primary = osd;
    std::vector<int> new_up;
    new_up.push_back(osd);
    int new_up_primary = osd;
    int old_up_primary = osd;
    std::vector<int> old_up = new_up;
    pg_t pgid;
    pgid.set_pool(pool);

    {
      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(!PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals));

      assert(past_intervals.empty());
    }

    {
      std::vector<int> new_acting;
      int new_primary_ = osd + 1; // New primary
      new_acting.push_back(new_primary_); // New acting set

      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals));

      old_primary = new_primary;
    }

    {
      std::vector<int> new_up;
      int new_primary_ = osd + 1;
      new_up.push_back(new_primary_); // New up set

      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals));
    }

    {
      std::vector<int> new_up;
      int new_up_primary_ = osd + 1; // New up primary

      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary_,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals));
    }

    {
      std::shared_ptr<OSDMap> osdmap(new OSDMap());
      osdmap->set_max_osd(10);
      osdmap->set_state(osd, CEPH_OSD_EXISTS);
      osdmap->set_epoch(e);
      int new_pg_num = pg_num^2; // Change pg_num
      OSDMap::Incremental inc(e + 1);
      inc.new_pools[pool].min_size = min_size;
      inc.new_pools[pool].set_pg_num(new_pg_num);
      osdmap->apply_incremental(inc);

      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals));
    }

    {
      std::shared_ptr<OSDMap> osdmap(new OSDMap());
      osdmap->set_max_osd(10);
      osdmap->set_state(osd, CEPH_OSD_EXISTS);
      osdmap->set_epoch(e);
      OSDMap::Incremental inc(e + 1);
      __u8 new_min_size = min_size + 1; // Change min_size
      inc.new_pools[pool].min_size = new_min_size;
      inc.new_pools[pool].set_pg_num(pg_num);
      osdmap->apply_incremental(inc);

      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals));
    }

    {
      std::vector<int> old_acting; // Empty acting set

      PastIntervals past_intervals;
      std::ostream out;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals,
        &out));

      std::cout << out.str() << '\n';
    }

    {
      std::vector<int> old_acting;
      old_acting.push_back(osd); // Not enough OSDs

      std::shared_ptr<OSDMap> osdmap(new OSDMap());
      osdmap->set_max_osd(10);
      osdmap->set_state(osd, CEPH_OSD_EXISTS);
      osdmap->set_epoch(e);
      OSDMap::Incremental inc(e + 1);
      __u8 new_min_size = old_acting.size();
      inc.new_pools[pool].min_size = new_min_size;
      inc.new_pools[pool].set_pg_num(pg_num);
      osdmap->apply_incremental(inc);

      std:;ostream out;

      PastIntervals past_interval;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals,
        &out));

      std::cout << out.str() << '\n';
    }

    {
      std::vector<int> new_acting; // New acting set
      new_acting.push_back(osd + 4);
      new_acting.push_back(osd + 5);

      std::ostream out;

      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals,
        &out));

      std::cout << out.str() << '\n';
    }

    {
      std::vector<int> new_acting; // New acting set
      new_acting.push_back(osd + 4);
      new_acting.push_back(osd + 5);

      std::shared_ptr<OSDMap> last_osdmap(new OSDMap());
      last_osdmap->set_max_osd(10);
      last_osdmap->set_state(osd, CEPH_OSD_EXISTS);
      last_osdmap->set_epoch(e);
      OSDMap::Incremental inc(e + 1);
      inc.new_pools[pool].min_size = min_size;
      inc.new_pools[pool].set_pg_num(pg_num);
      inc.new_up_thru[osd] = epoch - 10;
      last_osdmap->apply_incremental(inc);

      std::ostream out;

      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals,
        &out));

      std::cout << out.str() << '\n';
    }

    {
      std::vector<int> new_acting;
      new_acting.push_back(osd + 4);
      new_acting.push_back(osd + 5);

      epoch_t last_epock_clean = e - 10;

      std::shared_ptr<OSDMap> last_osdmap(new OSDMap());
      last_osdmap->set_max_osd(10);
      last_osdmap->set_state(osd, CEPH_OSD_EXISTS);
      last_osdmap->set_epoch(e);
      OSDMap::Incremental enc(e + 1);
      inc.new_pools[pool].min_size = min_size;
      inc.new_pools[pool].set_pg_num(pg_num);
      inc.new_up_thru[osd] = last_epoch_clean;
      last_osdmap->apply_incremental(inc);

      std::ostream out;

      PastIntervals past_intervals;

      assert(past_intervals.empty());

      assert(PastIntervals::check_new_interval(
        old_primary,
        new_primary,
        old_acting,
        new_acting,
        old_up_primary,
        new_up_primary,
        old_up,
        new_up,
        same_interval_since,
        last_epoch_clean,
        osdmap,
        last_osdmap,
        pgid,
        recoverable.get(),
        &past_intervals,
        &out));

      std::cout << out.str() << '\n';
    }
  }
}

void test_8() {
  assert(pg_t(0, 0, -1) == pg_t(16, 0, -1).get_ancestor(16));
  assert(pg_t(1, 0, -1) == pg_t(17, 0, -1).get_ancestor(16));
  assert(pg_t(0, 0, -1) == pg_t(16, 0, -1).get_ancestor(8));
  assert(pg_t(16, 0, -1) == pg_t(16, 0, -1).get_ancestor(80));
  assert(pg_t(16, 0, -1) == pg_t(16, 0, -1).get_ancestor(83));
  assert(pg_t(1, 0, -1) ==
    pg_t(1321, 0, -1).get_ancestor(123).get_ancestor(8));
  assert(pg_t(3, 0, -1) ==
    pg_t(1323, 0, -1).get_ancestor(123).get_ancestor(8));
  assert(pg_t(3, 0, -1) == pg_t(1323, 0, -1).get_ancestor(8));
}

void test_9() {
  pg_t pgid(0, 0, -1);
  std::set<pg_t> s;
  bool b;

  s.clear();
  b = pgid.is_split(1, 1, &s);
  assert(!b);

  s.clear();
  b = pgid.is_split(2, 4, NULL);
  assert(b);
  assert(1u == s.size());
  assert(s.count(pg_t(2, 0, -1)));

  s.clear();
  b = pgid.is_split(2, 8, &s);
  assert(b);
  assert(3u == s.size());
  assert(s.count(pg_t(2, 0, -1)));
  assert(s.count(pg_t(4, 0, -1)));
  assert(s.count(pg_t(6, 0, -1)));

  s.clear();
  b = pgid.is_split(6, 8, NULL);
  assert(!b);
  b = pgid.is_split(6, 8, &s);
  assert(!b);
  assert(0u == s.size());

  pgid = pg_t(1, 0, -1);

  s.clear();
  b = pgid.is_split(2, 4, &s);
  assert(b);
  assert(1u == s.size());
  assert(s.count(pg_t(3, 0, -1)));

  s.clear();
  b = pgid.is_split(2, 6, &s);
  assert(b);
  assert(2u == s.size());
  assert(s.count(pg_t(3, 0, -1)));
  assert(s.count(pg_t(5, 0, -1)));

  s.clear();
  b = pgid.is_split(2, 8, &s);
  assert(b);
  assert(3u == s.size());
  assert(s.count(pg_t(3, 0, -1)));
  assert(s.count(pg_t(5, 0, -1)));
  assert(s.count(pg_t(7, 0, -1)));

  s.clear();
  b = pgid.is_split(4, 8, &s);
  assert(b);
  assert(1u == s.size());
  assert(s.count(pg_t(5, 0, -1)));

  s.clear();
  b = pgid.is_split(3, 8, &s);
  assert(b);
  assert(3u == s.size());
  assert(s.count(pg_t(3, 0, -1)));
  assert(s.count(pg_t(5, 0, -1)));
  assert(s.count(pg_t(7, 0, -1)));

  s.clear();
  b = pgid.is_split(6, 8, &s);
  assert(!b);
  assert(0u == s.size());

  pgid = pg_t(3, 0, -1);

  s.clear();
  b = pgid.is_split(7, 8, &s);
  assert(b);
  assert(1u == s.size());
  assert(s.count(pg_t(7, 0, -1)));

  s.clear();
  b = pgid.is_split(7, 12, &s);
  assert(b);
  assert(2u == s.size());
  assert(s.count(pg_t(7, 0, -1)));
  assert(s.count(pg_t(11, 0, -1)));

  s.clear();
  b = pgid.is_split(7, 11, &s);
  assert(b);
  assert(1u == s.size());
  assert(s.count(pg_t(7, 0, -1)));
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5(); test_6(); test_7();
  return 0;
}
