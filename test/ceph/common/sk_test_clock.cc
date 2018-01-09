#include "common/Clock.h"
#include "common/ceph_time.h"
// #include "common/ceph_clock.h"
#include <iostream>

auto main() -> decltype(0) {
  utime_t c1 = ceph_clock_now();
  ceph::mono_time c2 = ceph::mono_clock::now();
  ceph::time_detail::coarse_mono_clock::time_point c3 = ceph::coarse_mono_clock::now();
  std::cout << c1 << '\n';
  std::cout << c2 << '\n';
  std::cout << c3 << '\n';
  return 0;
}
