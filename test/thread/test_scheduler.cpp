#include <boost/thread/executors/scheduler.hpp>
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/chrono/chrono_io.hpp>

#include <iostream>

void f(int x) {
  std::cout << __func__ << ": " << x << "\n";
}

struct test_scheduler {

  void after(boost::scheduler<>& s) {
    std::cout << __func__ << "\n";
    for (int i = 1; i <= 10; ++i) {
      s.after(boost::chrono::seconds(i)).submit(boost::bind(f, i));
      s.after(boost::chrono::milliseconds(i*100)).submit(boost::bind(f, i));
    }
  }

  void at(boost::scheduler<>& s) {
    std::cout << __func__ << "\n";
    for (int i = 1; i <= 10; ++i) {
      s.at(boost::chrono::steady_clock::now()+
        boost::chrono::seconds(i)).submit(boost::bind(f, i));
      s.at(boost::chrono::steady_clock::now()+
        boost::chrono::milliseconds(i*100)).submit(boost::bind(f, i));
    }
  }

  void on(boost::scheduler<>& s, boost::executors::basic_thread_pool& tp) {
    std::cout << __func__ << "\n";
    for (int i = 1; i <= 10; ++i) {
      s.on(tp).after(boost::chrono::seconds(i)).submit(boost::bind(f, i));
      s.on(tp).after(
        boost::chrono::milliseconds(i*100)).submit(boost::bind(f, i));
    }
  }

  void operator()(boost::scheduler<>& s,
    boost::executors::basic_thread_pool& tp) {
    after(s);
    at(s);
    on(s, tp);
    boost::this_thread::sleep_for(boost::chrono::seconds(20));
  }

};

void do_test_scheduler() {
  boost::executors::basic_thread_pool tp(4);
  boost::scheduler<> s;
  test_scheduler()(s, tp);
}

auto main() -> decltype(0) {
  do_test_scheduler();
  return 0;
}
