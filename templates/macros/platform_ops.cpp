#include "platform_selector.hpp"

template <typename T>
T doit() {
  platform_ops<platform_type> my_ops;
  my_ops.whoami();
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
