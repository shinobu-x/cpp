#include <primitiv/config.h>
#include <primitiv/naive_device.h>

#include <cassert>

void doit() {
  primitiv::devices::Naive dev1;
  primitiv::Device::set_default(dev1);
  assert(&dev1 == &primitiv::Device::get_default());

  primitiv::devices::Naive dev2;
  primitiv::Device::set_default(dev2);
  assert(&dev1 != &primitiv::Device::get_default());
  assert(&dev2 == &primitiv::Device::get_default());

  primitiv::devices::Naive dev3;
  primitiv::Device::set_default(dev3);
  assert(&dev1 != &primitiv::Device::get_default());
  assert(&dev2 != &primitiv::Device::get_default());
  assert(&dev3 == &primitiv::Device::get_default());
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
