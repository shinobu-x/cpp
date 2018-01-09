// #include "common/ceph_clock.h"

#include <iostream>
#include <thread>

void test_ceph_clock() {
  std::cout << __func__ << '\n';
/*
  ceph_clock cc;
  auto t1 = cc.now<utime_t>();
  auto t2 = cc.now<mono_time>();
  auto t3 = cc.now<coarse_mono_time>();
  auto t4 = cc.now<real_time>();
  auto t5 = cc.now<coarse_real_time>();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  auto e2 = cc.elapsed_from_start<mono_time>();
  auto e3 = cc.elapsed_from_start<coarse_mono_time>();
  auto e4 = cc.elapsed_from_start<real_time>();
  auto e5 = cc.elapsed_from_start<coarse_real_time>();
*/
/******
  auto t6 = cc.now<mono_time>();
  auto t7 = cc.now<coarse_mono_time>();
  auto t8 = cc.now<real_time>();
  auto t9 = cc.now<coarse_real_time>();
******/
/*
  ceph::mono_clock::duration ONE_SECOND_1;
  ceph::mono_clock::duration ONE_SECOND_2;
  cc.set_duration<seconds, coarse_mono_time>(ONE_SECOND_1, 1_s);
  cc.set_duration<seconds, coarse_mono_time>(ONE_SECOND_2, (int64_t)1);
*/
/******
  double c1 = std::chrono::duration_cast<seconds>(t6 - t2).count();
  double c2 = std::chrono::duration_cast<seconds>(t7 - t3).count();
  double c3 = std::chrono::duration_cast<seconds>(t8 - t4).count();
  double c4 = std::chrono::duration_cast<seconds>(t9 - t5).count();
  double c5 = std::chrono::duration_cast<milliseconds>(t6 - t2).count();
  double c6 = std::chrono::duration_cast<milliseconds>(t7 - t3).count();
  double c7 = std::chrono::duration_cast<milliseconds>(t8 - t4).count();
  double c8 = std::chrono::duration_cast<milliseconds>(t9 - t5).count();
******/
/*
  double c1 = cc.get_count<nanoseconds, coarse_real_time>();
  double c2 = cc.get_count<milliseconds, coarse_real_time>();
  double c3 = cc.get_count<seconds, coarse_real_time>();
  double c4 = cc.get_count<nanoseconds, coarse_mono_time>();
  double c5 = cc.get_count<milliseconds, coarse_mono_time>();
  double c6 = cc.get_count<seconds, coarse_mono_time>();

  std::cout << t1 << '\n';
  std::cout << t2 << '\n';
  std::cout << t3 << '\n';
  std::cout << t4 << '\n';
  std::cout << t5 << '\n';
  std::cout << e2 << '\n';
  std::cout << e3 << '\n';
  std::cout << e4 << '\n';
  std::cout << e5 << '\n';
  std::cout << c1 << '\n';
  std::cout << c2 << '\n';
  std::cout << c3 << '\n';
  std::cout << c4 << '\n';
  std::cout << c5 << '\n';
  std::cout << c6 << '\n';
*/
}

auto main() -> decltype(0) {
  test_ceph_clock(); 
}
