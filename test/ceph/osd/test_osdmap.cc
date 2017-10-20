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
  const static int num_osds = 0;
public:
  OSDMap osdmap;
  OSDMapMapping mapping;
  const uint64_t ec_pool = 1;
  const uint64_t rep_pool = 2;

  test_osdmap() {}

//  void set_up_map() {}
};

const static int num_osds = 6;

void set_up_map() {
  OSDMap osdmap;

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

}

TEST(test_osdmap, test_create) {
  set_up_map();
}
