#include <iostream>
#include <memory>
#include <gtest/gtest.h>

#include "include/stringify.h"
#include "crush/CrushWrapper.h"
#include "osd/osd_types.h"

#include <set>

std::unique_ptr<CrushWrapper> build_indep_map(CephContext* cct, int num_rack,
  int num_host, int num_osd) {
  std::unique_ptr<CrushWrapper> c(new CrushWrapper);
  c->create();

  c->set_type_name(5, "root");
  c->set_type_name(4, "row");
  c->set_type_name(3, "rack");
  c->set_type_name(2, "chasis");
  c->set_type_name(1, "host");
  c->set_type_name(0, "osd");

  int rootno;
  c->add_bucket(0, CRUSH_BUCKET_STRAW, CRUSH_HASH_RJENKINS1,
    5, 0, NULL, NULL, &rootno);

  c->set_item_name(rootno, "default");

  std::map<std::string, std::string> loc;

  loc["root"] = "default";

  int osd = 0;
  for (int r = 0; r < num_rack; ++r) {
    loc["rack"] = std::string("rack-") + stringify(r);
    for (int h = 0; h < num_host; ++h) {
      loc["host"] =
        std::string("host-") + stringify(r) + std::string("-") + stringify(h);
      for (int o = 0; o < num_osd; ++o, ++osd) {
        c->insert_item(
          cct, osd, 1.0, std::string("osd.") + stringify(osd), loc);
      }
    }
  }

  int ret;
  int ruleno = 0;
  ret = c->add_rule(ruleno, 4, 123, 1, 20);
  assert(ret == ruleno);
  ret = c->set_rule_step(ruleno, 0, CRUSH_RULE_SET_CHOOSELEAF_TRIES, 10, 0);
  assert(ret == 0);
  ret = c->set_rule_step(ruleno, 1, CRUSH_RULE_TAKE, rootno, 0);
  assert(ret == 0);
  ret =
    c->set_rule_step(ruleno, 2, CRUSH_RULE_CHOOSELEAF_INDEP, CRUSH_CHOOSE_N, 1);
  assert(ret == 0);
  ret = c->set_rule_step(ruleno, 3, CRUSH_RULE_EMIT, 0, 0);
  c->set_rule_name(ruleno, "data");

  c->finalize();

  if (false) {
    Formatter* f = Formatter::create("json-pretty");
    f->open_object_section("crush_map");
    c->dump(f);
    f->close_section();
    f->flush(cout);
    delete f;
  }
  return c;
}

int get_num_dups(const std::vector<int>& v) {
  std::set<int> s;
  int dups = 0;
  for (unsigned i = 0; i < v.size(); ++i)
    if (s.count(v[i]))
      ++dups;
    else if (v[i] != CRUSH_ITEM_NONE)
      s.insert(v[i]);
  return dups;
}

TEST(CRUSH, indep_too_small) {
  std::unique_ptr<CrushWrapper> c(build_indep_map(g_ceph_context, 1, 3, 1));
  std::vector<__u32> weight(c->get_max_devices(), 0x10000);
  c->dump_tree(&std::cout, NULL);

  for (int x = 0; x < 100; ++x) {
    std::vector<int> out;
    c->do_rule(0, x, out, 5, weight, 0);
    std::cout << x << " -> " << out << std::endl;
    int num_none = 0;
    for (unsigned i = 0; i < out.size(); ++i) {
      if (out[i] == CRUSH_ITEM_NONE)
        num_none++;
    }
    ASSERT_EQ(2, num_none);
    ASSERT_EQ(0, get_num_dups(out));
  }
}

TEST(CRUSH, indep_basic) {
  std::unique_ptr<CrushWrapper> c(build_indep_map(g_ceph_context, 3, 3, 3));
  vector<__u32> weight(c->get_max_devices(), 0x10000);
  c->dump_tree(&std::cout, NULL);

  for (int x = 0; x < 100; ++x) {
    std::vector<int> out;
    c->do_rule(0, x, out, 5, weight, 0);
    cout << x << " -> " << out << std::endl;
    int num_none = 0;
    for (unsigned i=0; i<out.size(); ++i) {
      if (out[i] == CRUSH_ITEM_NONE)
        num_none++;
    }
    ASSERT_EQ(0, num_none);
    ASSERT_EQ(0, get_num_dups(out));
  }
}
