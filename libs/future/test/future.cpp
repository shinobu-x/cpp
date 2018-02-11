#include "../include/futures.hpp"
#include "../hpp/future.hpp"

void doit() {
  {
    boost::BOOST_THREAD_FUTURE<int> f1;
    boost::BOOST_THREAD_FUTURE<boost::BOOST_THREAD_FUTURE<int> > f2;
    boost::future<int> f3;
    boost::future<boost::future<int> > f4;
  }
}

auto main() -> decltype(0) {
  return 0;
}
