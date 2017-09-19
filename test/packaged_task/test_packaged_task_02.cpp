#define BOOST_THREAD_VERSION 4

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <cassert>

int f() { return 42; }

boost::packaged_task<int()>* schedule(boost::function<int()> const& func) {
  boost::function<int()> copy(func);
  boost::packaged_task<int()>* r = new boost::packaged_task<int()>(copy);
  return r;
}

struct Func {
  Func(Func const&) = delete;
  Func& operator=(Func const&) = delete;
  Func() {};
  Func(Func&&) {};
  Func& operator=(Func&&) { return *this; };
  void operator()() const {}
};

auto main() -> decltype(0) {
  boost::packaged_task<int()>* task(schedule(f));
  (*task)();

  {
    boost::future<int> f = task->get_future();
    assert(f.get() == 42);
  }
  {
    boost::function<void()> f1;
    Func f2;

    boost::packaged_task<void()> task1(f1);
    boost::packaged_task<void()> task2(boost::move(f2));
  }

  return 0;
}
