#include <primitiv/config.h>
#include <primitiv/msgpack/objects.h>

#include <cassert>
#include <iostream>
#include <utility>

void doit() {
  {
    primitiv::msgpack::objects::Binary obj1;
    assert(!obj1.valid());

    try {
      obj1.check_valid();
    } catch (...) {}
    try {
      assert(obj1.size() == 0);
    } catch (...) {}
    try {
      assert(obj1.data() == nullptr);
    } catch (...) {}
    std::size_t s = 1024;
    auto r1 = obj1.allocate(s);
    auto r2 = obj1.size();
  }
}
auto main() -> decltype(0) {
  doit();
  return 0;
}
