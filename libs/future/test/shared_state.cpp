#include <include/futures.hpp>
#include <hpp/shared_state.hpp>

void doit() {
  {
    boost::detail::shared_state<int> s1;
    boost::detail::shared_state<void> s2;
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
