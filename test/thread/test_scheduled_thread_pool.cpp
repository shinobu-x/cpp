#include <boost/bind.hpp>
#include <boost/chrono.hpp>
#include <boost/chrono/chrono_io.hpp>
#include <boost/function.hpp>
#include <boost/thread/executors/scheduled_thread_pool.hpp>

#include <cassert>
#include <iostream>

void f1(int x) {
  std::cout << __func__ << ": " << x << '\n';
}

void f2(boost::chrono::steady_clock::time_point pushed,
  boost::chrono::steady_clock::duration d) {
  std::cout << __func__ << "\n";
  assert((pushed + d) < boost::chrono::steady_clock::now());
}

void f3(boost::scheduled_thread_pool* tp,
  boost::chrono::steady_clock::duration d) {
  boost::function<void()> f = boost::bind(f2,
    boost::chrono::steady_clock::now(), d);
  tp->submit_after(f, d);
}

struct timing {
  void operator()(const int n) {
    boost::scheduled_thread_pool tp(4);

    for (int i = 1; i <= n; ++i)
      tp.submit_after(boost::bind(f1, i), boost::chrono::milliseconds(i*100));

    boost::this_thread::sleep_for(boost::chrono::seconds(10));
  }
};

struct deque {
  void operator()() {
    boost::scheduled_thread_pool tp(4);

    for (int i = 0; i < 10; ++i) {
      typename boost::chrono::steady_clock::duration d =
        boost::chrono::milliseconds(i*100);
      boost::function<void()> f =
        boost::bind(f2, boost::chrono::steady_clock::now(), d);
      tp.submit_after(f, d);
    }
  }
};

struct deque_multi {
  void operator()(const int n) {
    boost::scheduled_thread_pool tp(4);
    boost::thread_group threads;

    for (int i = 0; i < n; ++i) {
      typename boost::chrono::steady_clock::duration d =
        boost::chrono::milliseconds(i*100);
      threads.create_thread(boost::bind(f3, &tp, d));
    }

    threads.join_all();
  }
};

void do_test_scheduled_thread_pool() {
  boost::chrono::steady_clock::time_point start =
    boost::chrono::steady_clock::now();

  timing()(5);

  boost::chrono::steady_clock::duration duration =
    boost::chrono::steady_clock::now() - start;

  assert(duration > boost::chrono::milliseconds(500));

  deque()();
  deque_multi()(4);
  deque_multi()(8);
  deque_multi()(16);
}

auto main() -> decltype(0) {
  do_test_scheduled_thread_pool();
  return 0;
}
