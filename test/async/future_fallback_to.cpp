#include <boost/thread/detail/log.hpp>
// #include <boost/thread/future.hpp>


#include <cassert>
#include <exception>
#include <future>
#include <string>
#include <iostream>

#include "../macro/config.hpp"

// #if defined BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION

int func_ex() {
  BOOST_THREAD_LOG << __func__ << BOOST_THREAD_END_LOG;
  throw std::logic_error("Error");
}

int func() {
  BOOST_THREAD_LOG << __func__ << BOOST_THREAD_END_LOG;
  return 1;
}

auto main() -> decltype(0) {
  const unsigned number_of_tests = 200;
  BOOST_THREAD_LOG << __func__ << BOOST_THREAD_END_LOG;

  {
    for (unsigned i = 0; i < number_of_tests; ++i)
      try {
        BOOST_THREAD_LOG << "" << BOOST_THREAD_END_LOG;
        std::future<int> f1 = std::async(std::launch::async, &func);
        BOOST_THREAD_LOG << "" << BOOST_THREAD_END_LOG;
        f1.wait();
        assert(f1.get() == 1);
        BOOST_THREAD_LOG << "" << BOOST_THREAD_END_LOG;
      } catch (std::exception& e) {
        LOG;
        std::cout << e.what() << '\n';
        return 1;
      } catch (...) {
        LOG;
        return 2;
      }
  }
       
  return 0;
}
/*
#else

auto main() -> decltype(0) {
  std::cout << __LINE__ << '\n';
  return 0;
}

#endif
*/
