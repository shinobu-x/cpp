#include "osd/PGLog.h"
#include "osd/OSDMap.h"
#include "include/coredumpctl.h"
#include "../../objectstore/store_test_fixture.h"

#include <stdio.h>
#include <signal.h>

#include <cassert>
#include <iostream>

#include "gtest/gtest.h"

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
  : protected PGLog, public base {
public:
  pglog_test() : PGLog(g_ceph_context) {}

#include "common/ceph_context.h"
#include "common/config.h"

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
  };
};

auto main() -> decltype(0) {
  return 0;
}
