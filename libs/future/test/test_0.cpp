#include "../hpp/future.hpp"
void doit() {
  boost::promise<int> p;
  auto f = p.get_future();
  int i = 1;
  p.set_value(i);
  auto r = f.get();
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
