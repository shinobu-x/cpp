#include "include/int_types.h"
#include "include/types.h"

#include "common/perf_counters.h"
#include "common/admin_socket_client.h"
#include "common/ceph_context.h"
#include "common/config.h"
#include "common/errno.h"
#include "common/safe_io.h"

#include "common/code_environment.h"
#include "global/global_context.h"
#include "global/global_init.h"
#include "include/msgr.h"

#include <errno.h>
#include <fcntl.h>
#include <map>
#include <poll.h>
#include <sstream>
#include <stdint.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "common/common_init.h"

enum {
  ELEMENT_FIRST = 200,
  ELEMENT_1,
  ELEMENT_2,
  ELEMENT_3,
  ELEMENT_LAST,
};

static PerfCounters* setup_test_perfcounters1(CephContext* cct) {
  PerfCountersBuilder b(cct, "test_perfcounters1", ELEMENT_1, ELEMENT_LAST);
  b.add_u64(ELEMENT_1, "element1");
  b.add_time(ELEMENT_2, "element2");
  b.add_time_avg(ELEMENT_3, "element3");
  return b.create_perf_counters();
}

void test_1() {
  AdminSocketClient c(get_rand_socket_path());
  std::string msg;
  c.do_request("{ \"prefix\": \"perf dump\" }", &msg);
}
/*
void test_2() {
  PerfCountersCollection* coll = g_ceph_context->perf_perfcounters_collection();
  PerfCounters* pf = setup_test_perfcounters1(CephContext* cct);
  AdminSocketClient c(get_rand_socket_path());
}
*/
auto main() -> decltype(0) {
  test_1();
  return 0;
}
