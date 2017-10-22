#include "gtest/gtest.h"
#include "osd/OSDMap.h"
#include "osd/OSDMapMapping.h"

#include "global/global_context.h"
#include "global/global_init.h"
#include "common/common_init.h"
#include "common/ceph_argparse.h"

#include <iostream>

auto main(int argc, char** argv) -> decltype(0) {
  std::vector<const char*> args(argv, argv+argc);
  env_to_vec(args);
  auto cct = global_init(nullptr, args, CEPH_ENTITY_TYPE_CLIENT,
    CODE_ENVIRONMENT_UTILITY, CINIT_FLAG_NO_DEFAULT_CONFIG_FILE);
  common_init_finish(g_ceph_context);
  g_ceph_context->_conf->set_val("osd_pool_default_size", "3", false);
  g_ceph_context->_conf->set_val("osd_crush_chooseleaf_type", "0", false);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

class test_osdmap : public testing::Test {
private:
const static int num_osds = 6;

public:
  OSDMap osdmap;
  OSDMapMapping mapping;
  const uint64_t ec_pool = 1;
  const uint64_t rep_pool = 2;

  test_osdmap() {}

  void set_up_map() {
    uuid_d fsid;
    const uint64_t ec_pool = 1;
    const uint64_t rep_pool = 2;

    osdmap.build_simple(g_ceph_context, 0, fsid, num_osds);
    OSDMap::Incremental pending_inc(osdmap.get_epoch() + 1);
    pending_inc.fsid = osdmap.get_fsid();
    entity_addr_t test_addr;
    uuid_d test_uuid;

    for (int i = 0; i < num_osds; ++i) {
      test_uuid.generate_random();
      test_addr.nonce = i;
      pending_inc.new_state[i] = CEPH_OSD_EXISTS | CEPH_OSD_NEW;
      pending_inc.new_up_client[i] = test_addr;
      pending_inc.new_up_cluster[i] = test_addr;
      pending_inc.new_hb_back_up[i] = test_addr;
      pending_inc.new_hb_front_up[i] = test_addr;
      pending_inc.new_weight[i] = CEPH_OSD_IN;
      pending_inc.new_uuid[i] = test_uuid;
    }
    osdmap.apply_incremental(pending_inc);

    int crush_rule = osdmap.crush->add_simple_rule("erasure", "default", "osd",
      "", "indep", pg_pool_t::TYPE_ERASURE, &std::cerr);

    OSDMap::Incremental new_pool_inc(osdmap.get_epoch() + 1);
    new_pool_inc.new_pool_max = osdmap.get_pool_max();
    new_pool_inc.fsid = osdmap.get_fsid();
    pg_pool_t empty;

    // EC
    uint64_t pool_id = ++new_pool_inc.new_pool_max;
    assert(pool_id == ec_pool);
    pg_pool_t* p = new_pool_inc.get_new_pool(pool_id, &empty);
    p->size = 3;
    p->set_pg_num(64);
    p->set_pgp_num(64);
    p->type = pg_pool_t::TYPE_ERASURE;
    p->crush_rule = crush_rule;
    new_pool_inc.new_pool_names[pool_id] = "ec";

    // Replicated
    pool_id = ++new_pool_inc.new_pool_max;
    assert(pool_id == rep_pool);
    p = new_pool_inc.get_new_pool(pool_id, &empty);
    p->size = 3;
    p->set_pg_num(64);
    p->set_pgp_num(64);
    p->type = pg_pool_t::TYPE_REPLICATED;
    p->crush_rule = 0;
    p->set_flag(pg_pool_t::FLAG_HASHPSPOOL);
    new_pool_inc.new_pool_names[pool_id] = "rep";
    osdmap.apply_incremental(new_pool_inc);
  }

  unsigned int get_num_osds() {
    return num_osds;
  }

  void test_mappings(int pool, int num, std::vector<int>* any,
    std::vector<int>* first, std::vector<int>* primary) {
    mapping.update(osdmap);

    for (int i = 0; i < num; ++i) {
      std::vector<int> up, acting;
      int up_primary, acting_primary;
      pg_t pgid(i, pool);
      osdmap.pg_to_up_acting_osds(
        pgid, &up, &up_primary, &acting, &acting_primary);

      for (unsigned j = 0; j < acting.size(); ++j)
        (*any)[acting[j]]++;
      if (!acting.empty())
        (*first)[acting[0]]++;
      if (acting_primary >= 0)
        (*primary)[acting_primary]++;

      std::vector<int> up2, acting2;
      int up_primary2, acting_primary2;
      pgid = osdmap.raw_pg_to_pg(pgid);
      mapping.get(pgid, &up2, &up_primary2, &acting2, &acting_primary2);
      ASSERT_EQ(up, up2);
      ASSERT_EQ(up_primary, up_primary2);
      ASSERT_EQ(acting, acting2);
      ASSERT_EQ(acting_primary, acting_primary2);
    }

    std::cout << "any: " << *any << std::endl;
    std::cout << "first: " << *first << std::endl;
    std::cout << "primary: " << *primary << std::endl;
  }
};

TEST_F(test_osdmap, create) {
  set_up_map();
  ASSERT_EQ(get_num_osds(), (unsigned)osdmap.get_max_osd());
  ASSERT_EQ(get_num_osds(), osdmap.get_num_in_osds());
}

TEST_F(test_osdmap, features) {
  set_up_map();
  uint64_t features = osdmap.get_features(CEPH_ENTITY_TYPE_OSD, NULL);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES2);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES3);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_V2);
  ASSERT_TRUE(features & CEPH_FEATURE_OSD_ERASURE_CODES);
  ASSERT_TRUE(features & CEPH_FEATURE_OSDHASHPSPOOL);
  ASSERT_TRUE(features & CEPH_FEATURE_OSD_PRIMARY_AFFINITY);

  features = osdmap.get_features(CEPH_ENTITY_TYPE_CLIENT, NULL);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES2);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES3);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_V2);
  ASSERT_TRUE(features & CEPH_FEATURE_OSDHASHPSPOOL);
  ASSERT_TRUE(features & CEPH_FEATURE_OSD_PRIMARY_AFFINITY);
  ASSERT_FALSE(features & CEPH_FEATURE_OSD_ERASURE_CODES);

  {
    OSDMap::Incremental new_pool_inc(osdmap.get_epoch() + 1);
    new_pool_inc.old_pools.insert(osdmap.lookup_pg_pool_name("ec"));
    new_pool_inc.new_primary_affinity[0] = 0x8000;
    osdmap.apply_incremental(new_pool_inc);
  }

  features = osdmap.get_features(CEPH_ENTITY_TYPE_MON, NULL);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES2);
  ASSERT_TRUE(features & CEPH_FEATURE_CRUSH_TUNABLES3);
  ASSERT_TRUE(features & CEPH_FEATURE_OSDHASHPSPOOL);
  ASSERT_TRUE(features & CEPH_FEATURE_OSD_PRIMARY_AFFINITY);
  ASSERT_FALSE(features & CEPH_FEATURE_CRUSH_V2);
  ASSERT_FALSE(features & CEPH_FEATURE_OSD_ERASURE_CODES);
}

TEST_F(test_osdmap, map_pg) {
  set_up_map();

  std::cerr << " osdmap.pool_max == " << osdmap.get_pool_max() << std::endl;
  pg_t raw_pg(0, rep_pool, -1);
  pg_t pgid = osdmap.raw_pg_to_pg(raw_pg);
  std::vector<int> up_osds, acting_osds;
  int up_primary, acting_primary;

  osdmap.pg_to_up_acting_osds(
    pgid, &up_osds, &up_primary, &acting_osds, &acting_primary);

  std::vector<int> old_up_osds, old_acting_osds;
  osdmap.pg_to_up_acting_osds(pgid, old_up_osds, old_acting_osds);
  ASSERT_EQ(old_up_osds, up_osds);
  ASSERT_EQ(old_acting_osds, acting_osds);
  ASSERT_EQ(osdmap.get_pg_pool(rep_pool)->get_size(), up_osds.size());
}

TEST_F(test_osdmap, map_function) {
  set_up_map();
  pg_t raw_pg(0, rep_pool, -1);
  pg_t pgid = osdmap.raw_pg_to_pg(raw_pg);
  std::vector<int> up_osds, acting_osds;
  int up_primary, acting_primary;

  osdmap.pg_to_up_acting_osds(
    pgid, &up_osds, &up_primary, &acting_osds, &acting_primary);

  std::vector<int> up_osds_two, acting_osds_two;

  osdmap.pg_to_up_acting_osds(pgid, up_osds_two, acting_osds_two);

  ASSERT_EQ(up_osds, up_osds_two);
  ASSERT_EQ(acting_osds, acting_osds_two);

  int acting_primary_two;
  osdmap.pg_to_acting_osds(pgid, &acting_osds_two, &acting_primary_two);
  EXPECT_EQ(acting_osds, acting_osds_two);
  EXPECT_EQ(acting_primary, acting_primary_two);
  osdmap.pg_to_acting_osds(pgid, acting_osds_two);
  EXPECT_EQ(acting_osds, acting_osds_two);
}

TEST_F(test_osdmap, primary) {
  set_up_map();
  pg_t raw_pg(0, rep_pool, -1);
  pg_t pgid = osdmap.raw_pg_to_pg(raw_pg);
  std::vector<int> up_osds, acting_osds;
  int up_primary, acting_primary;

  osdmap.pg_to_up_acting_osds(
    pgid, &up_osds, &up_primary, &acting_osds, &acting_primary);

  std::vector<int> up_osds_two, acting_osds_two;

  osdmap.pg_to_up_acting_osds(
    pgid, up_osds_two, acting_osds_two);

  ASSERT_EQ(up_osds, up_osds_two);
  ASSERT_EQ(acting_osds, acting_osds_two);

  int acting_primary_two;

  osdmap.pg_to_acting_osds(
    pgid, &acting_osds_two, &acting_primary_two);

  EXPECT_EQ(acting_osds, acting_osds_two);
  EXPECT_EQ(acting_primary, acting_primary_two);

  osdmap.pg_to_acting_osds(pgid, acting_osds_two);

  EXPECT_EQ(acting_osds, acting_osds_two);
}

TEST_F(test_osdmap, pg_temp_respected) {
  set_up_map();

  pg_t raw_pg(0, rep_pool, -1);
  pg_t pgid = osdmap.raw_pg_to_pg(raw_pg);
  std::vector<int> up_osds, acting_osds;
  int up_primary, acting_primary;

  osdmap.pg_to_up_acting_osds(
    pgid, &up_osds, &up_primary, &acting_osds, &acting_primary);

  // Non-primary OSD to primary via incremental
  OSDMap::Incremental pgtemp_map(osdmap.get_epoch() + 1);
  pgtemp_map.new_primary_temp[pgid] = acting_osds[1];
  osdmap.apply_incremental(pgtemp_map);

  osdmap.pg_to_up_acting_osds(
    pgid, &up_osds, &up_primary, &acting_osds, &acting_primary);

  EXPECT_EQ(acting_primary, acting_osds[1]);
  std::cout << acting_primary << std::endl;
  std::cout << acting_osds[1] << std::endl;
}

TEST_F(test_osdmap, clean_temps) {
  set_up_map();

  OSDMap::Incremental pgtemp_map(osdmap.get_epoch() + 1);
  OSDMap::Incremental pending_inc(osdmap.get_epoch() + 2);

  pg_t pg_a = osdmap.raw_pg_to_pg(pg_t(0, rep_pool));
  {
    std::vector<int> up_osds, acting_osds;
    int up_primary, acting_primary;
    osdmap.pg_to_up_acting_osds(
      pg_a, &up_osds, &up_primary, &acting_osds, &acting_primary);
    pgtemp_map.new_pg_temp[pg_a] =
      mempool::osdmap::vector<int>(up_osds.begin(), up_osds.end());
    pgtemp_map.new_primary_temp[pg_a] = up_primary;
  }

  pg_t pg_b = osdmap.raw_pg_to_pg(pg_t(1, rep_pool));
  {
    std::vector<int> up_osds, acting_osds;
    int up_primary, acting_primary;
    osdmap.pg_to_up_acting_osds(
      pg_b, &up_osds, &up_primary, &acting_osds, &acting_primary);
    pending_inc.new_pg_temp[pg_b] =
      mempool::osdmap::vector<int>(up_osds.begin(), up_osds.end());
    pending_inc.new_primary_temp[pg_b] = up_primary;
  }

  osdmap.apply_incremental(pgtemp_map);
  OSDMap::clean_temps(g_ceph_context, osdmap, &pending_inc);

  EXPECT_TRUE(pending_inc.new_pg_temp.count(pg_a) &&
    pending_inc.new_pg_temp[pg_a].size() == 0);
  EXPECT_EQ(-1, pending_inc.new_primary_temp[pg_a]);
  EXPECT_TRUE(!pending_inc.new_pg_temp.count(pg_b) &&
    !pending_inc.new_primary_temp.count(pg_b));
}

TEST_F(test_osdmap, necessary_temp) {
  set_up_map();
  pg_t raw_pg(0, rep_pool, -1);
  pg_t pgid = osdmap.raw_pg_to_pg(raw_pg);
  std::vector<int> up_osds, acting_osds;
  int up_primary, acting_primary;

  osdmap.pg_to_up_acting_osds(
    pgid, &up_osds, &up_primary, &acting_osds, &acting_primary);

  OSDMap::Incremental pgtemp_map(osdmap.get_epoch() + 1);

  int i = 0;
  for (; i != (int)get_num_osds(); ++i) {
    bool in_use = false;

    for (std::vector<int>::iterator osd = up_osds.begin();
      osd != up_osds.end(); ++osd) {
      if (i == *osd) {
        in_use = true;
        break;
      }
    }

    if (!in_use) {
      up_osds[1] = i;
      break;
    }
  }

  if (i == (int)get_num_osds())
    FAIL() << "Unused OSD Not Found";


  pgtemp_map.new_pg_temp[pgid] =
    mempool::osdmap::vector<int>(up_osds.begin(), up_osds.end());
  pgtemp_map.new_primary_temp[pgid] = up_osds[1];
  osdmap.apply_incremental(pgtemp_map);

  OSDMap::Incremental pending_inc(osdmap.get_epoch() + 1);

  OSDMap::clean_temps(g_ceph_context, osdmap, &pending_inc);
  EXPECT_FALSE(pending_inc.new_pg_temp.count(pgid));
  EXPECT_FALSE(pending_inc.new_primary_temp.count(pgid));
}

TEST_F(test_osdmap, primary_affinity) {
  set_up_map();

  int n = get_num_osds();

  for (std::map<int64_t, pg_pool_t>::const_iterator p =
    osdmap.get_pools().begin(); p != osdmap.get_pools().end(); ++p) {
    int pool = p->first;
    int expect_primary = 10000 / n;

    std::cout << "pool " << pool << " size " << (int)p->second.size
      << " expect_primary" << expect_primary << std::endl;

    {
      std::vector<int> any(n, 0);
      std::vector<int> first(n, 0);
      std::vector<int> primary(n, 0);
      test_mappings(pool, 10000, &any, &first, &primary);

      for (int i = 0; i < n; ++i) {
        ASSERT_LT(0, any[i]);
        ASSERT_LT(0, first[i]);
        ASSERT_LT(0, primary[i]);
      }
    }

    osdmap.set_primary_affinity(0, 0);
    osdmap.set_primary_affinity(1, 0);

    {
      std::vector<int> any(n, 0);
      std::vector<int> first(n, 0);
      std::vector<int> primary(n, 0);
      test_mappings(pool, 10000, &any, &first, &primary);

      for (int i = 0; i < n; ++i) {
        ASSERT_LT(0, any[i]);
        if (i >= 2) {
          ASSERT_LT(0, first[i]);
          ASSERT_LT(0, primary[i]);
        } else {
          if (p->second.is_replicated()) {
            ASSERT_EQ(0, first[i]);
          }
          ASSERT_EQ(0, primary[i]);
        }
      }
    }

    osdmap.set_primary_affinity(0, 0x8000);
    osdmap.set_primary_affinity(1, 0);

    {
      std::vector<int> any(n, 0);
      std::vector<int> first(n, 0);
      std::vector<int> primary(n, 0);
      test_mappings(pool, 10000, &any, &first, &primary);

      int expect = (10000 / (n-2)) / 2;
      std::cout << "expect " << expect << std::endl;

      for (int i = 0; i < n; ++i) {
        ASSERT_LT(0, any[i]);
        if (i >= 2) {
          ASSERT_LT(0, first[i]);
          ASSERT_LT(0, primary[i]);
        } else if (i == 1) {
          if (p->second.is_replicated()) {
            ASSERT_EQ(0, first[i]);
          }
          ASSERT_EQ(0, primary[i]);
        } else {
          ASSERT_LT(expect * 2/3, primary[0]);
          ASSERT_GT(expect * 4/3, primary[0]);
        }
      }
    }
    osdmap.set_primary_affinity(0, 0x10000);
    osdmap.set_primary_affinity(1, 0x10000);
  }
}
