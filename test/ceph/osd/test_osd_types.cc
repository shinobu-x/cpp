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
  assert(prefixes_correct == prefixes_out);
}

void test_4() {
  uint32_t mask = 0xE947FA20;
  uint32_t bits = 32;
  int64_t pool = 0x23;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000023.02AF749E"));
  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));
  assert(prefixes_correct == prefixes_out);
}

void test_5() {
  uint32_t mask = 0xE947FA20;
  uint32_t bits = 0;
  int64_t pool = 0x23;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000023."));

  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));
  assert(prefixes_correct == prefixes_out);
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
  assert(prefixes_correct == prefixes_out);
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
      std::ostringstream out;

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
        &past_intervals,
        &out));

      assert(past_intervals.empty());

      std::cout << "= CASE1 ================================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
    }

    {
      std::vector<int> new_acting;
      int new_primary_ = osd + 1; // New primary
      new_acting.push_back(new_primary_); // New acting set

      std::ostringstream out;

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

      old_primary = new_primary;

      std::cout << "= CASE2 ================================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
    }

    {
      std::vector<int> new_up;
      int new_primary_ = osd + 1;
      new_up.push_back(new_primary_); // New up set

      std::ostringstream out;

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

      std::cout << "= CASE3 ===============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
    }

    {
      std::vector<int> new_up;
      int new_up_primary_ = osd + 1; // New up primary

      std::ostringstream out;

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
        &past_intervals,
        &out));

      std::cout << "= CASE4 ===============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
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

      std::ostringstream out;

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

      std::cout << "= CASE5 ===============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
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

      std::ostringstream out;

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

      std::cout << "= CASE6 ===============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
    }

    {
      std::vector<int> old_acting; // Empty acting set

      PastIntervals past_intervals;
      std::ostringstream out;

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

      std::cout << "= CASE7 ===============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
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

      std::ostringstream out;

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

      std::cout << "= CASE8 ===============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
    }

    {
      std::vector<int> new_acting; // New acting set
      new_acting.push_back(osd + 4);
      new_acting.push_back(osd + 5);

      std::ostringstream out;

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

      std::cout << "= CASE9 ===============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
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
      inc.new_up_thru[osd] = e - 10;
      last_osdmap->apply_incremental(inc);

      std::ostringstream out;

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

      std::cout << "= CASE10 ==============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
    }

    {
      std::vector<int> new_acting;
      new_acting.push_back(osd + 4);
      new_acting.push_back(osd + 5);

      epoch_t last_epoch_clean = e - 10;

      std::shared_ptr<OSDMap> last_osdmap(new OSDMap());
      last_osdmap->set_max_osd(10);
      last_osdmap->set_state(osd, CEPH_OSD_EXISTS);
      last_osdmap->set_epoch(e);
      OSDMap::Incremental inc(e + 1);
      inc.new_pools[pool].min_size = min_size;
      inc.new_pools[pool].set_pg_num(pg_num);
      inc.new_up_thru[osd] = last_epoch_clean;
      last_osdmap->apply_incremental(inc);

      std::ostringstream out;

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

      std::cout << "= CASE11 ==============================================\n";
      std::cout << out.str() << '\n';
      std::cout << "=======================================================\n";
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
  b = pgid.is_split(2, 4, &s);
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

// Test pg_missing_t
void test_10() {

  { // constructor
    pg_missing_t missing;
    assert((unsigned int)0 == missing.num_missing());
    assert(!missing.have_missing());
  }

  { // have_missing
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    pg_missing_t missing;
    assert(!missing.have_missing());
    missing.add(oid, eversion_t(), eversion_t(), false);
    assert(missing.have_missing());
  }

  { // claim
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    pg_missing_t missing;
    assert(!missing.have_missing());
    missing.add(oid, eversion_t(), eversion_t(), false);
    assert(missing.have_missing());

    pg_missing_t other;
    assert(!other.have_missing());

    other.claim(missing);
    assert(other.have_missing());
  }

  { // is_missing
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    pg_missing_t missing;
    assert(!missing.is_missing(oid));
    missing.add(oid, eversion_t(), eversion_t(), false);
    assert(missing.is_missing(oid));
  }

  { // is_missing
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    pg_missing_t missing;
    eversion_t need(10, 5);
    assert(!missing.is_missing(oid, eversion_t()));
    missing.add(oid, need, eversion_t(), false);
    assert(missing.is_missing(oid));
    assert(!missing.is_missing(oid, eversion_t()));
    assert(missing.is_missing(oid, need));
  }

  { // have_old
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    pg_missing_t missing;
    assert(eversion_t() == missing.have_old(oid));
    missing.add(oid, eversion_t(), eversion_t(), false);
    assert(eversion_t() == missing.have_old(oid));
    eversion_t have(1, 1);
    missing.revise_have(oid, have);
    assert(have == missing.have_old(oid));
  }

  { // add_next_event
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    hobject_t other(object_t("other"), "key", 987, 654, 0, "");
    eversion_t version(10, 5);
    eversion_t prior_version(3, 4);
    pg_log_entry_t sample(pg_log_entry_t::DELETE, oid, version, prior_version,
      0, osd_reqid_t(entity_name_t::CLIENT(777), 8, 999), utime_t(8, 9), 0);

    {  // new object
      pg_missing_t missing;
      pg_log_entry_t e = sample;

      e.op = pg_log_entry_t::MODIFY;
      e.prior_version = eversion_t();
      assert(e.is_update());
      assert(e.object_is_indexed());
      assert(e.reqid_is_indexed());
      assert(!missing.is_missing(oid));
      missing.add_next_event(e);
      assert(missing.is_missing(oid));
      assert(eversion_t() == missing.get_items().at(oid).have);
      assert(oid == missing.get_rmissing().at(e.version.version));
      assert(1U == missing.num_missing());
      assert(1U == missing.get_rmissing().size());

      missing.add_next_event(e);
      assert(missing.is_missing(oid));
      assert(1U == missing.num_missing());
      assert(1U == missing.get_rmissing().size());
    }

    { // new object - clone
      pg_missing_t missing;
      pg_log_entry_t e = sample;

      e.op = pg_log_entry_t::CLONE;
      e.prior_version = eversion_t();
      assert(e.is_clone());
      assert(e.object_is_indexed());
      assert(!e.reqid_is_indexed());
      assert(!missing.is_missing(oid));
      missing.add_next_event(e);
      assert(missing.is_missing(oid));
      assert(eversion_t() == missing.get_items().at(oid).have);
      assert(oid == missing.get_rmissing().at(e.version.version));
      assert(1U == missing.num_missing());
      assert(1U == missing.get_rmissing().size());

      missing.add_next_event(e);
      assert(missing.is_missing(oid));
      assert(1U == missing.num_missing());
      assert(1U == missing.get_rmissing().size());
    }

    { // existing object - modify
      pg_missing_t missing;
      pg_log_entry_t e = sample;

      e.op = pg_log_entry_t::MODIFY;
      e.prior_version = eversion_t();
      assert(e.is_update());
      assert(e.object_is_indexed());
      assert(e.reqid_is_indexed());
      assert(!missing.is_missing(oid));
      missing.add_next_event(e);
      assert(missing.is_missing(oid));
      assert(eversion_t() == missing.get_items().at(oid).have);
      assert(oid == missing.get_rmissing().at(e.version.version));
      assert(1U == missing.num_missing());
      assert(1U == missing.get_rmissing().size());

      e.prior_version = prior_version;
      missing.add_next_event(e);
      assert(eversion_t() == missing.get_items().at(oid).have);
      assert(missing.is_missing(oid));
      assert(1U == missing.num_missing());
      assert(1U == missing.get_rmissing().size());
    }

    { // object with prior version - modify
      pg_missing_t missing;
      pg_log_entry_t e = sample;

      e.op = pg_log_entry_t::MODIFY;
      assert(e.is_update());
      assert(e.object_is_indexed());
      assert(e.reqid_is_indexed());
      assert(!missing.is_missing(oid));
      missing.add_next_event(e);
      assert(missing.is_missing(oid));
      assert(prior_version == missing.get_items().at(oid).have);
      assert(version == missing.get_items().at(oid).need);
      assert(oid == missing.get_rmissing().at(e.version.version));
      assert(1U == missing.num_missing());
      assert(1U == missing.get_rmissing().size());
    }

    { // backlog
      pg_missing_t missing;
      pg_log_entry_t e = sample;
      e.op = pg_log_entry_t::BACKLOG;
      assert(e.is_backlog());
      assert(e.object_is_indexed());
      assert(!e.reqid_is_indexed());
      assert(!missing.is_missing(oid));
    }

    {
      pg_missing_t missing;
      pg_log_entry_t e = sample;

      e.op = pg_log_entry_t::MODIFY;
      assert(e.is_update());
      assert(e.object_is_indexed());
      assert(e.reqid_is_indexed());
      assert(!missing.is_missing(oid));
      missing.add_next_event(e);
      assert(missing.is_missing(oid));
      assert(!missing.get_items().at(oid).is_delete());

      e.op = pg_log_entry_t::LOST_DELETE;
      e.version.version++;
      assert(e.is_delete());
      missing.add_next_event(e);
      assert(missing.is_missing(oid));
      assert(missing.get_items().at(oid).is_delete());
      assert(prior_version == missing.get_items().at(oid).have);
      assert(e.version == missing.get_items().at(oid).need);
      assert(oid == missing.get_rmissing().at(e.version.version));
      assert(1U == missing.num_missing());
      assert(1U == missing.get_rmissing().size());
    }
  }

  { // revise_need
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    pg_missing_t missing;
    assert(!missing.is_missing(oid));
    eversion_t need(10, 10);
    missing.revise_need(oid, need, false);
    assert(missing.is_missing(oid));
    assert(eversion_t() == missing.get_items().at(oid).have);
    assert(need == missing.get_items().at(oid).need);
    eversion_t have(1, 1);
    missing.revise_have(oid, have);
    eversion_t new_need(10, 12);
    assert(have == missing.get_items().at(oid).have);
    missing.revise_need(oid, new_need, false);
    assert(have == missing.get_items().at(oid).have);
    assert(new_need == missing.get_items().at(oid).need);
  }

  { // revise_have
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    pg_missing_t missing;
    assert(!missing.is_missing(oid));
    eversion_t have(1, 1);
    missing.revise_have(oid, have);
    assert(!missing.is_missing(oid));
    eversion_t need(10, 12);
    missing.add(oid, need, have, false);
    assert(missing.is_missing(oid));
    eversion_t new_have(2, 2);
    assert(have == missing.get_items().at(oid).have);
    missing.revise_have(oid, new_have);
    assert(new_have == missing.get_items().at(oid).have);
    assert(need == missing.get_items().at(oid).need);
  }

  { // add
    hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
    pg_missing_t missing;
    assert(!missing.is_missing(oid));
    eversion_t have(1, 1);
    eversion_t need(10, 10);
    missing.add(oid, need, have, false);
    assert(missing.is_missing(oid));
    assert(have == missing.get_items().at(oid).have);
    assert(need == missing.get_items().at(oid).need);
  }

  {
    { // rm
      hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
      pg_missing_t missing;
      assert(!missing.is_missing(oid));
      epoch_t e = 10;
      eversion_t need(e, 10);
      missing.add(oid, need, eversion_t(), false);
      assert(missing.is_missing(oid));
      missing.rm(oid, eversion_t(e / 2, 20));
      assert(missing.is_missing(oid));
      missing.rm(oid, eversion_t(e * 2, 20));
      assert(!missing.is_missing(oid));
    }

    {
      hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
      pg_missing_t missing;
      assert(!missing.is_missing(oid));
      missing.add(oid, eversion_t(), eversion_t(), false);
      assert(missing.is_missing(oid));
      std::map<hobject_t, pg_missing_item>::const_iterator m =
        missing.get_items().find(oid);
      missing.rm(m);
      assert(!missing.is_missing(oid));
    }
  }

  { // got
    {
      hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
      pg_missing_t missing;

      assert(!missing.is_missing(oid));
      epoch_t e = 10;
      eversion_t need(e, 10);
      missing.add(oid, need, eversion_t(), false);
      assert(missing.is_missing(oid));

      missing.got(oid, eversion_t(e * 2, 20));
      assert(!missing.is_missing(oid));

    }

    {
      hobject_t oid(object_t("test"), "key", 123, 456, 0, "");
      pg_missing_t missing;
      assert(!missing.is_missing(oid));
      missing.add(oid, eversion_t(), eversion_t(), false);
      assert(missing.is_missing(oid));
      std::map<hobject_t, pg_missing_item>::const_iterator m =
        missing.get_items().find(oid);
      missing.got(m);
      assert(!missing.is_missing(oid));
    } 
  }

  { // split_into
    uint32_t hash1 = 1, hash2 = 2;
    hobject_t oid1(object_t("test"), "key1", 123, hash1, 0, "");
    hobject_t oid2(object_t("test"), "key2", 123, hash2, 0, "");
    pg_missing_t missing;
    missing.add(oid1, eversion_t(), eversion_t(), false);
    missing.add(oid2, eversion_t(), eversion_t(), false);
    pg_t child_pgid;
    child_pgid.m_seed = 1;
    pg_missing_t child;
    unsigned split_bits = 1;
    missing.split_into(child_pgid, split_bits, &child);
    assert(child.is_missing(oid1));
    assert(!child.is_missing(oid2));
    assert(!missing.is_missing(oid1));
    assert(missing.is_missing(oid2));
  }

  { // clear
    hobject_t oid1(object_t("test"), "key1", 123, 1, 0, "");
    hobject_t oid2(object_t("test"), "key2", 123, 2, 0, "");
    pg_missing_t missing;
    missing.add(oid1, eversion_t(), eversion_t(), false);
    missing.add(oid2, eversion_t(), eversion_t(), false);
    assert(missing.is_missing(oid1));
    assert(missing.is_missing(oid2));
    missing.clear();
  }

  { // dump
    hobject_t oid1(object_t("test"), "key1", 123, 1, 0, "");
    hobject_t oid2(object_t("test"), "key2", 123, 1, 0, "");
    pg_missing_t missing;
    missing.add(oid1, eversion_t(), eversion_t(), false);
    missing.add(oid2, eversion_t(), eversion_t(), false);
    Formatter* f = Formatter::create("json-pretty");
    missing.dump(f);
  }

  { // test
    pg_missing_t missing;
    missing.test();
  }
}

class object_context {
public:
  static const useconds_t DELAY_MAX = 20 * 1000 * 1000;

  class read_lock : public Thread {
  public:
    ObjectContext& obc;

    explicit read_lock(ObjectContext& _obc)
      : obc(_obc) {}

    void* entry() override {
      obc.ondisk_read_lock();
      return NULL;
    }
  };

  class write_lock : public Thread {
  public:
    ObjectContext& obc;

    explicit write_lock(ObjectContext& _obc)
      : obc(_obc) {}

    void* entry() override {
      obc.ondisk_write_lock();
      return NULL;
    }
  };
};

void test_11() {
  {
    ObjectContext obc;
  
    // no lock
    assert(0 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);  

    // lock: 1
    obc.ondisk_write_lock();
    assert(0 == obc.writers_waiting);
    assert(1 == obc.unstable_writes);

    // lock: 2
    obc.ondisk_write_lock();
    assert(0 == obc.writers_waiting);
    assert(2 == obc.unstable_writes);

    // unlock: 1
    obc.ondisk_write_unlock();
    assert(0 == obc.writers_waiting);
    assert(1 == obc.unstable_writes);

    // unlock: 2
    obc.ondisk_write_unlock();
    assert(0 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);
  }

  useconds_t delay = 0;

  { // read lock
    ObjectContext obc;

    assert(0 == obc.readers_waiting);
    assert(0 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);

    // lock
    obc.ondisk_write_lock();
    assert(0 == obc.readers_waiting);
    assert(0 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(1 == obc.unstable_writes);

    object_context::read_lock t(obc);
    t.create("obc_read");

    do {
      std::cout << "Trying... " << delay << '\n';
      usleep(delay);
    } while (obc.readers_waiting == 0 &&
      (delay = delay * 2 + 1) < object_context::DELAY_MAX);

    assert(1 == obc.readers_waiting);
    assert(0 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(1 == obc.unstable_writes);

    obc.ondisk_write_unlock();

    do {
      std::cout << "Trying... " << delay << '\n';
      usleep(delay);
    } while ((obc.readers == 0 || obc.readers_waiting == 1) &&
      (delay = delay * 2 + 1) < object_context::DELAY_MAX);

    assert(0 == obc.readers_waiting);
    assert(1 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);

    obc.ondisk_read_unlock();

    assert(0 == obc.readers_waiting);
    assert(0 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);

    t.join();
  }

  { // write lock
    ObjectContext obc;

    assert(0 == obc.readers_waiting);
    assert(0 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);

    obc.ondisk_read_lock();
    assert(0 == obc.readers_waiting);
    assert(1 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);

    object_context::write_lock t(obc);
    t.create("obc_write");

    do {
      std::cout << "Trying... " << delay << '\n';
      usleep(delay);
    } while ((obc.writers_waiting == 0) &&
      (delay = delay * 2 + 1) < object_context::DELAY_MAX);

    assert(0 == obc.readers_waiting);
    assert(1 == obc.readers);
    assert(1 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);

    obc.ondisk_read_unlock();

    do {
      std::cout << "Trying... " << delay << '\n';
      usleep(delay);
    } while ((obc.unstable_writes == 0 || obc.writers_waiting == 1) &&
      (delay == delay * 2 + 1) < object_context::DELAY_MAX);

    assert(0 == obc.readers_waiting);
    assert(0 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(1 == obc.unstable_writes);

    obc.ondisk_write_unlock();

    assert(0 == obc.readers_waiting);
    assert(0 == obc.readers);
    assert(0 == obc.writers_waiting);
    assert(0 == obc.unstable_writes);

    t.join();
  }
}

void test_12() {

  { // get_pg_num_divisor
    pg_pool_t p;
    p.set_pg_num(16);
    p.set_pgp_num(16);

    for (int i = 0; i < 16; ++i)
      assert(16u == p.get_pg_num_divisor(pg_t(i, 1)));

    p.set_pg_num(12);
    p.set_pgp_num(12);

    assert(16u == p.get_pg_num_divisor(pg_t(0, 1)));
    assert(16u == p.get_pg_num_divisor(pg_t(1, 1)));
    assert(16u == p.get_pg_num_divisor(pg_t(2, 1)));
    assert(16u == p.get_pg_num_divisor(pg_t(3, 1)));
    assert(8u == p.get_pg_num_divisor(pg_t(4, 1)));
    assert(8u == p.get_pg_num_divisor(pg_t(5, 1)));
    assert(8u == p.get_pg_num_divisor(pg_t(6, 1)));
    assert(8u == p.get_pg_num_divisor(pg_t(7, 1)));
    assert(16u == p.get_pg_num_divisor(pg_t(8, 1)));
    assert(16u == p.get_pg_num_divisor(pg_t(9, 1)));
    assert(16u == p.get_pg_num_divisor(pg_t(10, 1)));
    assert(16u == p.get_pg_num_divisor(pg_t(11, 1)));
  }

  { // get_random_pg_position
    srand(getpid());

    for (int i = 0; i < 100; ++i) {
      pg_pool_t p;
      p.set_pg_num(1 + (rand() % 1000));
      p.set_pgp_num(p.get_pg_num());
      pg_t pgid(rand() % p.get_pg_num(), 1);
      uint32_t h = p.get_random_pg_position(pgid, rand());
      uint32_t ps = p.raw_hash_to_pg(h);
      std::cout << "pg_num: " << p.get_pg_num() << " " 
        << "pgid: " << pgid << " psition: " << h
        << " -> " << pg_t(ps, 1) << '\n';
      assert(pgid.ps() == ps);
    } 
  }

  { // generate_test_instances
    std::list<pg_info_t*> i;
    i.push_back(new pg_info_t);
    i.push_back(new pg_info_t);
    std::list<pg_history_t*> h;
    pg_history_t::generate_test_instances(h);
    i.back()->history = *h.back();
    i.back()->pgid = spg_t(pg_t(1, 2, -1), shard_id_t::NO_SHARD);
    i.back()->last_update = eversion_t(3, 4);
    i.back()->last_complete = eversion_t(5, 6);
    i.back()->last_user_version = 2;
    i.back()->log_tail = eversion_t(7, 8);
    i.back()->last_backfill =
      hobject_t(object_t("test"), "key", 123, 456, -1, "");
    i.back()->last_backfill_bitwise = true;
    {
      std::list<pg_stat_t*> s;
      pg_stat_t::generate_test_instances(s);
      i.back()->stats = *s.back();
    }
    {
      std::list<pg_hit_set_history_t*> s;
      pg_hit_set_history_t::generate_test_instances(s);
      i.back()->hit_set = *s.back();
    }
    {
      std::list<pg_info_t*> i;
      pg_info_t::generate_test_instances(i);
    }
  }
}

void test_13() {
  { // shard_id_t
    std::set<shard_id_t> shards;
    shards.insert(shard_id_t(0));
    shards.insert(shard_id_t(1));
    shards.insert(shard_id_t(2));
    std::ostringstream out;
    out << shards;
    std::cout << out.str() << '\n';
    shard_id_t noshard = shard_id_t::NO_SHARD;
    shard_id_t zero(0);
    assert(zero >= noshard);
  }

  { // spg_t::parse
    spg_t a(pg_t(1, 2), shard_id_t::NO_SHARD);
    spg_t aa, bb;
    spg_t b(pg_t(3, 2), shard_id_t(2));
    std::string s = stringify(a);
    assert(aa.parse(s.c_str()));
    assert(a == aa);

    s = stringify(b);
    assert(bb.parse(s.c_str()));
    assert(b == bb);
  }

  { // coll_t::parse
    const char* ok[] = {
      "meta",
      "1.2_head",
      "1.2_TEMP",
      "1.2s3_head",
      "1.3s2_TEMP",
      "1.2s0_head",
      0
    };

    const char* bad[] = {
      "foo",
      "1.2_food",
//      "1.2_head",
//      "1.2_temp",
//      "1.2_HEAD",
//      "1.xS3_HEAD",
//      "1.2s_HEAD",
      "1.2sfoo_HEAD",
      0
    };

    coll_t a;
    for (int i = 0; ok[i]; ++i) {
      std::cout << "ok" << ok[i] << '\n';
      assert(a.parse(ok[i]));
      assert(std::string(ok[i]) == a.to_str());
    }

    for (int i = 0; bad[i]; ++i) {
      std::cout << "bad" << bad[i] << '\n';
      assert(!a.parse(bad[i])); 
    }  
  }

  { // coll_t
    {
      spg_t pgid1;
      coll_t foo(pgid1);
      assert(foo.to_str() == std::string("0.0_head"));

      coll_t temp = foo.get_temp();
      assert(temp.to_str() == std::string("0.0_TEMP"));

      spg_t pgid2;
      assert(temp.is_temp());
      assert(temp.is_temp(&pgid2));
      assert(pgid1 == pgid2);
    }

    {
      spg_t pgid;
      coll_t right(pgid);
      assert(right.to_str() == std::string("0.0_head"));
      std::cout << right.to_str() << '\n';

      coll_t left, middle;
      assert(left.to_str() == std::string("meta"));
      assert(middle.to_str() == std::string("meta"));

      left = middle = right;

      assert(left.to_str() == std::string("0.0_head"));
      std::cout << left.to_str() << '\n';

      assert(middle.to_str() == std::string("0.0_head"));
      std::cout << middle.to_str() << '\n';

      assert(middle.c_str() != right.c_str());
      assert(left.c_str() != middle.c_str());
    }
  }
}

void test_14() {
  { // hobject_t
    const char* v[] = {
      "MIN",
      "MAX",
      "-1:60c2fa6d:::inc_osdmap.1:0",
      "-1:60c2fa6d:::inc_osdmap.1:333",
      "0:00000000::::head",
      "1:00000000:nspace:key:obj:head",
      "-40:00000000:nspace:obj:head",
      "20:00000000::key:obj:head",
      "20:00000000:::o%fdj:head",
      "20:00000000:::o%02fdj:head",
      "20:00000000:::_zero_%00_:head",
      NULL
    };

    for (unsigned i = 0; v[i]; ++i) {
      hobject_t o;
      bool b = o.parse(v[i]);

      if (!b)
        std::cout << "failed to parse " << v[i] << std::endl;

      std::string s = stringify(o);

      if (s != v[i])
        std::cout << v[i] << " -> " << o << " -> " << s << std::endl;
    }
  }
}

void test_15() {
  { // objectstore_perf_stat_t
    std::list<objectstore_perf_stat_t*> o;
    o.push_back(new objectstore_perf_stat_t());
    o.push_back(new objectstore_perf_stat_t());
    o.back()->os_commit_latency = 20;
    o.back()->os_apply_latency = 30;
    Formatter* f = Formatter::create("json-pretty");
    o.back()->dump(f);
  }
}

void test_16() {
  { // ghobject_t
    {
      ghobject_t min, sep;
      sep.set_shard(shard_id_t(1));
      sep.hobj.pool = -1;
      std::cout << min << " <- " << sep << '\n';
      assert(min < sep);

      sep.set_shard(shard_id_t::NO_SHARD);
      std::cout << "sep shard" << sep.shard_id << '\n';

      ghobject_t o(hobject_t(
        object_t(), std::string(), CEPH_NOSNAP, 0x42, 1, std::string()));
      std::cout << "o " << o << '\n';
      assert(o > sep);
    }

    { 
      const char* v[] = {
        "GHMIN",
        "GHMAX",
        "13#0:00000000::::head#",
        "13#0:00000000::::head#deadbeef",
        "#-1:60c2fa6d:::inc_osdmap.1:333#deadbeef",
        "#-1:60c2fa6d:::inc%02osdmap.1:333#deadbeef",
        "#-1:60c2fa6d:::inc_osdmap.1:333#",
        "1#MIN#deadbeefff",
        "1#MAX#",
        "#MAX#123",
        "#-40:00000000:nspace::obj:head#",
        NULL
      };

      for (const char* c : v) 
        std::cout << c << '\n';

      for (unsigned i = 0; v[i]; ++i) {
        ghobject_t o;
        bool b = o.parse(v[i]);

        if (!b)
          std::cout << "failed to parse " << v[i] << '\n';

        std::string s = stringify(o);

        if (s != v[i])
          std::cout << v[i] << " -> " << o << " -> " << s << '\n';
      }
    }
  }
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5(); test_6(); test_7();
  test_8(); test_9(); test_10(); test_11(); test_12(); test_13(); test_14();
  test_15(); test_16();
  return 0;
}
