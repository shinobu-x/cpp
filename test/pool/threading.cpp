#include <boost/pool/pool_alloc.hpp>
#include <boost/thread.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

void do_test() {
  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(-10, 10);
  std::list<unsigned, boost::fast_pool_allocator<unsigned> > l;

  for (unsigned i = 0; i < 100; ++i)
    l.push_back(i);

  for (unsigned i = 0; i < 100000; ++i) {
    int val = dist(gen);
    if (val < 0) {
      while (val && l.size()) {
        l.pop_back();
        ++i;
      }
    } else {
      while (val) {
        l.push_back(val);
        --val;
      }
    }
  }
}

auto main() -> decltype(0) {
  std::list<boost::shared_ptr<boost::thread> > threads;
  for (unsigned i = 0; i < 10; ++i)
    try {
      threads.push_back(boost::shared_ptr<boost::thread>(
        new boost::thread(&do_test)));
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
    }

  std::list<boost::shared_ptr<
    boost::thread> >::const_iterator a(threads.begin()), b(threads.end());

  while (a != b) {
    (*a)->join();
    ++a;
  }
  return 0;
}
