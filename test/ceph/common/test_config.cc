#include "common/ceph_argparse.h"
#include "common/config.h"
#include "global/global_context.h"

#include <iostream>
#include <string>

void test_1() {
  std::cout << 
    g_ceph_context->_conf->public_addr << '\n';
}
auto main() -> decltype(0) {
  test_1();
  return 0;
}
