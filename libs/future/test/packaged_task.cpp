#include <include/futures.hpp>
#include <hpp/packaged_task.hpp>

int a() {
  std::cout << __func__ << '\n';
  return 1;
}

void invoke_lazy_task(boost::packaged_task<int()>& task) {
  std::cout << __func__ << '\n';
  try {
    task();
  } catch (...) {}
}

void doit() {
  boost::packaged_task<int()> task(a);
  task.set_wait_callback(invoke_lazy_task);
  auto f(task.get_future());
  assert(!f.is_ready());
  assert(!f.has_value());
  auto r = f.get();
  assert(f.is_ready());
  assert(f.has_value());
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
