#include <include/futures.hpp>
#include <hpp/packaged_task.hpp>

int a() {
  return 1;
}

void invoke_lazy_task(boost::packaged_task<int()>& task) {
  try {
    task();
  } catch (...) {}
}

void doit() {
  boost::packaged_task<int()> task(a);
  task.set_wait_callback(invoke_lazy_task);
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
