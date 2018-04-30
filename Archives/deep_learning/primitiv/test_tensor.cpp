#include <primitiv/primitiv.h>
#include "../utils/utils.hpp"

#include <cassert>
#include <iostream>

void doit() {
  {
    const primitiv::Tensor t;
    assert(!t.valid());
    try {
      t.shape();
      t.device();
      t.to_float();
      t.to_vector();
    } catch (...) {}
  }
  {
    std::vector<primitiv::Device*> dev;
    add_available_devices(dev);
    auto it = *dev.begin();
    const primitiv::Tensor t = it->new_tensor_by_constant({}, 1);
    assert(t.valid());
    assert(it == &t.device());
    assert(primitiv::Shape() == t.shape());
    assert(1.0f == t.to_float());
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
