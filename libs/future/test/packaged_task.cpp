#include <include/futures.hpp>
// #include <hpp/packaged_task.hpp>
// #include "/usr/include/boost/thread/future.hpp"
boost::condition_variable cv;
boost::mutex m;
int ans = 0;
bool is_ready = false;

int a() {
  std::cout << __func__ << '\n';
  return 1;
}

void b() {
  std::cout << __func__ << '\n';
  boost::unique_lock<boost::mutex> lock(m);
  boost::this_thread::sleep_for(boost::chrono::seconds(5));
  ans = 1;
  is_ready = true;
}

void c() {
  std::cout << __func__ << '\n';
  boost::unique_lock<boost::mutex> lock(m);
  while (!is_ready) {
    cv.wait(lock);
  }
  ans = 2;
}

void invoke_lazy_task(boost::packaged_task<int>& task) {
  std::cout << __func__ << '\n';
  try {
    task();
  } catch (...) {}
}

void doit() {
  {
    boost::packaged_task<int> task(a);
    task.set_wait_callback(invoke_lazy_task);
     auto f(task.get_future());
    assert(!f.is_ready());
    assert(!f.has_value());
    auto r = f.get();
    assert(f.is_ready());
    assert(f.has_value());
  }
  {
    b();
    c();
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
