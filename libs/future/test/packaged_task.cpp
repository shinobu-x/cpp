#include <include/futures.hpp>

int a() {
  std::cout << boost::this_thread::get_id() << '\n';
  return 1;
}

void doit() {
  boost::packaged_task<int()> task(a);
  auto f = task.get_future();
  boost::thread th(boost::move(task));
  f.wait();
  assert(f.is_ready());
  assert(f.has_value());
  assert(!f.has_exception());
  assert(f.get_state() == boost::future_state::ready);
  assert(f.get() == 1);
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
