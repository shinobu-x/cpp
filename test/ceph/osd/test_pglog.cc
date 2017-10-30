#include <stdio.h>
#include <signal.h>
#include "gtest/gtest.h"
#include "osd/PGLog.h"
#include "osd/OSDMap.h"
#include "include/coredumpctl.h"
#include "../../objectstore/store_test_fixture.h"

struct base {
  static hobject_t mk_obj(unsigned id) {
    hobject_t hoid;
    std::stringstream ss;
    ss << "obj_" << id;
    hoid.oid = ss.str();
    hoid.set_hash(id);
    hoid.pool = 1;
    return hoid;
  }

  static eversion_t mk_evt(unsigned e, unsigned v) {
    return eversion_t(e, v);
  }

  static pg_log_entry_t mk_ple_mod(const hobject_t& hoid, eversion_t v,
    eversion_t pv, osd_reqid_t reqid) {
    pg_log_entry_t e;
    e.mark_unrollbackable();
    e.op = pg_log_entry_t::MODIFY;
    e.soid = hoid;
    e.version = v;
    e.prior_version = pv;
    e.reqid = reqid;
    return e;
  }

  static pg_log_entry_t mk_ple_dt(const hobject_t& hoid, eversion_t v,
    eversion_t pv, osd_reqid_t reqid) {
    pg_log_entry_t e;
    e.mark_unrollbackable();
    e.op = pg_log_entry_t::LOST_DELETE;
    e.soid = hoid;
    e.version = v;
    e.prior_version = pv;
    e.reqid = reqid;
    return e;
  }

  static pg_log_entry_t mk_ple_ldt(const hobject_t& hoid, eversion_t v,
    eversion_t pv) {
    pg_log_entry_t e;
    e.mark_unrollbackable();
    e.op = pg_log_entry_t::LOST_DELETE;
    e.soid = hoid;
    e.version = v;
    e.prior_version = pv;
    return e;
  }

  static pg_log_entry_t mk_ple_mod_rb(const hobject_t& hoid, eversion_t v,
    eversion_t pv, osd_reqid_t reqid) {
    pg_log_entry_t e;
    e.op = pg_log_entry_t::MODIFY;
    e.soid = hoid;
    e.version = v;
    e.prior_version = v;
    e.reqid = reqid;
    return e;
  }

  static pg_log_entry_t mk_ple_dt_rb(const hobject_t& hoid, eversion_t v,
    eversion_t pv, osd_reqid_t reqid) {
    pg_log_entry_t e;
    e.op = pg_log_entry_t::DELETE;
    e.soid = hoid;
    e.version = v;
    e.prior_version = pv;
    e.reqid = reqid;
    return e;
  }

  static pg_log_entry_t mk_ple_err(const hobject_t& hoid, eversion_t v,
    osd_reqid_t reqid) {
    pg_log_entry_t e;
    e.op = pg_log_entry_t::ERROR;
    e.soid = hoid;
    e.version = v;
    e.prior_version = eversion_t(0, 0);
    e.reqid = reqid;
    return e;
  }

  static pg_log_entry_t mk_ple_mod(const hobject_t& hoid, eversion_t v,
    eversion_t pv) {
    return mk_ple_mod(hoid, v, pv, osd_reqid_t());
  }

  static pg_log_entry_t mk_ple_dt(const hobject_t& hoid, eversion_t v,
    eversion_t pv) {
    return mk_ple_mod_rb(hoid, v, pv, osd_reqid_t());
  }

  static pg_log_entry_t mk_ple_dt_rb(const hobject_t& hoid, eversion_t v,
    eversion_t pv) {
    return mk_ple_dt_rb(hoid, v, pv, osd_reqid_t());
  }

  static pg_log_entry_t mk_ple_err(const hobject_t& hoid, eversion_t v) {
    return mk_ple_err(hoid, v, osd_reqid_t());
  }
};

class pglog_test
  : public ::testing::Test, protected PGLog, public base {
public:
  pglog_test() : PGLog(g_ceph_context) {}

  void SetUp() override {
    missing.may_include_deletes = true;
  }

#include "common/ceph_context.h"
#include "common/config.h"

  void TearDown() override {
    clear();
  }

  struct test {
    std::list<pg_log_entry_t> base;
    std::list<pg_log_entry_t> auth;
    std::list<pg_log_entry_t> div;

    pg_missing_t init;
    pg_missing_t last;

    std::set<hobject_t> to_remove;
    std::list<pg_log_entry_t> to_rollback;
    bool deletes_during_peering;

  private:
    IndexedLog full_auth;
    IndexedLog full_div;
    pg_info_t auth_info;
    pg_info_t div_info;

  public:
    test() : deletes_during_peering(false) {}

    void setup() {
      init.may_include_deletes = !deletes_during_peering;
      last.may_include_deletes = !deletes_during_peering;
      full_auth.log.insert(full_auth.log.end(), base.begin(), base.end());
      full_auth.log.insert(full_auth.log.end(), auth.begin(), auth.end());
      full_div.log.insert(full_div.log.end(), base.begin(), base.end());
      full_div.log.insert(full_div.log.end(), div.begin(), div.end());

      full_auth.head = auth_info.last_update =
        full_auth.log.rbegin()->version;
      auth_info.last_complete = full_auth.log.rbegin()->version;
      auth_info.log_tail = full_auth.log.begin()->version;
      auth_info.log_tail.version--;
      full_auth.tail = auth_info.log_tail;
      auth_info.last_backfill = hobject_t::get_max();

      if (init.get_items().empty())
        div_info.last_complete = div_info.last_update;
      else {
        eversion_t fmissing =
          init.get_items().at(init.get_rmissing().begin()->second).need;

        for (std::list<pg_log_entry_t>::const_iterator i =
          full_div.log.begin(); i != full_div.log.end(); ++i)
          if (i->version < fmissing)
            div_info.last_complete = i->version;
          else
            break;
      }
    }
    void set_div_bounds(eversion_t head, eversion_t tail) {
      full_auth.tail = auth_info.log_tail = tail;
      full_auth.head = auth_info.last_update = head;
    }

    const IndexedLog& get_full_auth() const {
      return full_auth;
    }
    const IndexedLog& get_full_div() const {
      return full_div;
    }
    const pg_info_t& get_auth_info() const {
      return auth_info;
    }
    const pg_info_t& get_div_info() const {
      return div_info;
    }
  };

  struct log_handler : public PGLog::LogEntryHandler {
    std::set<hobject_t> removed;
    std::list<pg_log_entry_t> rolledback;

    void rollback(const pg_log_entry_t& entry) override {
      rolledback.push_back(entry);
    }

    void rollforward(const pg_log_entry_t& entry) override {}

    void remove(const hobject_t& hoid) override {
      removed.insert(hoid);
    }

    void try_stash(const hobject_t&, version_t) override {}

    void trim(const pg_log_entry_t& entry) override {}
  };

  template <typename missing_t>
  void verify_missing(const test& t, const missing_t& missing) {

    assert(t.last.get_items().size() == missing.get_items().size());

    for (auto i = missing.get_items().begin();
      i != missing.get_items().end(); ++i) {
      assert(t.last.get_items().count(i->first));
      assert(t.last.get_items().find(i->first)->second.need ==
        i->second.need);
      assert(t.last.get_items().find(i->first)->second.have ==
        i->second.have);
    }

    bool correct = missing.debug_verify_from_init(t.init, &std::cout);
    assert(correct);
  }

  void verify_side_effects(const test& t, const log_handler& h) {
    assert(t.to_remove.size() == h.removed.size());
    assert(t.to_rollback.size() == h.rolledback.size());

    {
      std::list<pg_log_entry_t>::const_iterator t_iter = t.to_rollback.begin();
      std::list<pg_log_entry_t>::const_iterator h_iter = h.rolledback.begin();

      for (; t_iter != t.to_rollback.end(); ++t_iter, ++h_iter)
        assert(t_iter->version == h_iter->version);
    }

    {
      std::set<hobject_t>::const_iterator t_iter = t.to_remove.begin();
      std::set<hobject_t>::const_iterator h_iter = h.removed.begin();

      for (; t_iter != t.to_remove.end(); ++t_iter, ++h_iter)
        assert(*t_iter == *h_iter);
    }
  }

  void test_merge_log(const test& t) {
    clear();

    log = t.get_full_div();
    pg_info_t info = t.get_div_info();

    missing = t.init;
    missing.flush();

    IndexedLog olog;
    olog = t.get_full_auth();
    pg_info_t oinfo = t.get_auth_info();

    log_handler h;
    bool dirty_info = false;
    bool dirty_big_info = false;

    merge_log(oinfo, olog, pg_shard_t(1, shard_id_t(0)), info, &h,
      dirty_info, dirty_big_info);

    assert(info.last_update == oinfo.last_update);

    verify_missing(t, missing);
    verify_side_effects(t, h);
  }

  void test_proc_replica_log(const test& t) {
    clear();

    log = t.get_full_auth();
    pg_info_t info = t.get_auth_info();

    pg_missing_t omissing = t.init;

    IndexedLog olog;
    olog = t.get_full_div();
    pg_info_t oinfo = t.get_div_info();

    proc_replica_log(oinfo, olog, omissing, pg_shard_t(1, shard_id_t(0)));

    assert(oinfo.last_update >= log.tail);

    if (!t.base.empty())
      assert(t.base.rbegin()->version == oinfo.last_update);

    for (std::list<pg_log_entry_t>::const_iterator i = t.auth.begin();
      i != t.auth.end(); ++i)
      if (i->version > oinfo.last_update) {
        if (i->is_delete() && t.deletes_during_peering)
          omissing.rm(i->soid, i->version);
        else
          omissing.add_next_event(*i);
      }
    verify_missing(t, omissing);
  }

  void do_test(const test& t) {
    test_merge_log(t);
    test_proc_replica_log(t);
  }
};

struct test_handler : public PGLog::LogEntryHandler {
  std::list<hobject_t>& _removed;

  explicit test_handler(std::list<hobject_t>& removed) : _removed(removed) {}

  void rollback(const pg_log_entry_t& entry) override {}

  void rollforward(const pg_log_entry_t& entry) override {}

  void remove(const hobject_t& hoid) override {
    _removed.push_back(hoid);
  }

  void cant_rollback(const pg_log_entry_t& entry) {}

  void try_stash(const hobject_t&, version_t) override {}

  void trim(const pg_log_entry_t& entry) override {}
};

TEST_F(pglog_test, rewind_divergent_log) {
  {
    clear();
    pg_info_t info;
    std::list<hobject_t> remove_snap;
    bool dirty_info = false;
    bool dirty_big_info = false;

    hobject_t divergent_object;
    eversion_t divergent_version;
    eversion_t new_head;

    hobject_t divergent;
    divergent.set_hash(0x09);

    {
      pg_log_entry_t e;
      e.mark_unrollbackable();

      e.version = eversion_t(1, 1);
      e.soid.set_hash(0x5);
      log.tail = e.version;
      log.log.push_back(e);
      e.version = new_head = eversion_t(1, 4);
      e.soid = divergent;
      log.log.push_back(e);
      e.version = divergent_version = eversion_t(1, 5);
      e.prior_version = eversion_t(1, 4);
      e.soid = divergent;
      divergent_object = e.soid;
      e.op = pg_log_entry_t::DELETE;
      log.log.push_back(e);
      log.head = e.version;
      log.index();

      info.last_update = log.head;
      info.last_complete = log.head;
    }

    assert(!missing.have_missing());
    assert(3U == log.log.size());
    assert(remove_snap.empty());
    assert(log.head == info.last_update);
    assert(log.head == info.last_complete);
    assert(!is_dirty());
    assert(!dirty_info);
    assert(!dirty_big_info);

    test_handler h(remove_snap);
    rewind_divergent_log(new_head, info, &h, dirty_info, dirty_big_info);
  }
}

