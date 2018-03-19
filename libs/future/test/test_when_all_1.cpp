#include <include/futures.hpp>
#include <stdexcept>

auto a() {
  return 1;
}

auto b() {
  throw std::logic_error("1");
}

void doit() {
  {
    boost::future<int> f = boost::make_ready_future(1);
    assert(!f.valid());
    assert(f.is_ready());
    boost::future<boost::csbl::tuple<boost::future<int> > > fs =
      boost::when_all(boost::move(f));
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
