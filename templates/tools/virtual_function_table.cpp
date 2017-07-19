#include "hpp/virtual_function_table.hpp"

template <typename T>
T doit() {
  type_t<T> type;
  type.do_some_a();
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
