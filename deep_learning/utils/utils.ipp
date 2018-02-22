#ifndef TEST_UTILS_IPP
#define TEST_UTILS_IPP

#include <primitiv/primitiv.h>
#include <vector>
void add_device(std::vector<primitiv::Device*> &devs, primitiv::Device* dev) {
  devs.emplace_back(dev);
  dev->dump_description();
}

void add_available_devices(std::vector<primitiv::Device*> &dev) {
  add_device(dev, new primitiv::devices::Naive());
}

#endif // TEST_UTILS_IPP
