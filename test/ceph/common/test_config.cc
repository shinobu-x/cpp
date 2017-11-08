#include "common/ceph_argparse.h"
#include "common/config.h"
#include "global/global_context.h"

#include <iostream>
#include <string>

void test_1() {
  CephContext* cct;
  auto r = cct->_conf->get_val<std::string>("osd_pool_default_erasure_code_profile");
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
