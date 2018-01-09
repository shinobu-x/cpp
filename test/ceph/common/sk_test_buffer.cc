#include "include/buffer.h"
#include "common/ceph_time.h"

#include <iostream>

void test_buffer() {
  std::cout << __func__ << '\n';
  ceph::time_detail::coarse_mono_clock::time_point t1
    = ceph::time_detail::coarse_mono_clock::now();

  char zeros[128] = {0};

  bufferptr bp = buffer::create_static(sizeof(zeros), &zeros[0]);
  bufferlist bl;

  for (int i = 1; i <= 4096; ++i)
    bl.append(bp);

  for (int i = 1; i <= 10000; ++i)
    bl.zero();

  ceph::time_detail::coarse_mono_clock::time_point t2
    = ceph::time_detail::coarse_mono_clock::now();

  auto c1 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);

  std::cout << c1 << '\n';
}

auto main() -> decltype(0) {
  test_buffer();
  return 0;
}
