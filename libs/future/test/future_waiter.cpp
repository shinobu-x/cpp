#include <include/futures.hpp>
#include <hpp/future_waiter.hpp>

void doit() {
  {
    boost::detail::future_waiter fw;
    boost::future<int> f;
    fw.add(f);
  }
}
auto main() -> decltype(0) {
  return 0;
}
