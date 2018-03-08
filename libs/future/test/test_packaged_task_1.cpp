#include <include/futures.hpp>

template <typename F>
boost::future<typename boost::result_of<F()>::type> spawn_task(F f) {
  typedef typename boost::result_of<F()>::type result_type;
  boost::packaged_task<result_type()> task(boost::move(f));
  boost::future<result_type>  r(task.get_future());
  boost::thread(boost::move(task));
  return r;
}

int a() {
  return 1;
}

auto main() -> decltype(0) {
  auto f = spawn_task(a);
  f.wait();
  assert(f.is_ready());
  return 0;
}
