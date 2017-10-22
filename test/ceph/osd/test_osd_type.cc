#include "include/types.h"
#include "osd/osd_types.h"
#include "osd/OSDMap.h"
#include "gtest/gtest.h"
#include "common/Thread.h"
#include "include/stringify.h"
#include "osd/ReplicatedBackend.h"

#include <iostream>

void test_1() {
  uint32_t mask = 0xE947FA20;
  uint32_t bits = 12;
  int64_t pool = 0;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000000.02A"));
  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));
  std::cout << prefixes_correct << std::endl;
  std::cout << prefixes_out << std::endl;
}

void test_2() {
  uint32_t mask = 0x0000000F;
  uint32_t bits = 6;
  int64_t pool = 20;

  std::set<std::string> prefixes_correct;
  prefixes_correct.insert(std::string("0000000000000014.F0"));
  prefixes_correct.insert(std::string("0000000000000014.F4"));
  prefixes_correct.insert(std::string("0000000000000014.F8"));
  prefixes_correct.insert(std::string("0000000000000014.FC"));

  std::set<std::string> prefixes_out(
    hobject_t::get_prefixes(bits, mask, pool));
  std::cout << prefixes_correct << std::endl;
  std::cout << prefixes_out << std::endl;
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}
