#include <include/futures.hpp>

void doit() {
  {
    boost::promise<int> p1;
    boost::future<int> f1 = p1.get_future();
    bool thrown = false;
    try {
      f1 = p1.get_future();
    } catch(boost::future_already_retrieved const&) {
      thrown = true;
    }
    assert(thrown);
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
