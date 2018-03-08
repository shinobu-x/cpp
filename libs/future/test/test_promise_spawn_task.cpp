#include <include/futures.hpp>

template <typename F>
boost::future<typename boost::result_of<F()>::type> spawn_task(F f) {
  typedef typename boost::result_of<F()>::type result_type;

  struct task {
    boost::promise<result_type> p;
    F f_;
    task(task const& that) = delete;
    task(F f) : f_(f) {}
    task(task&& that) : p(boost::move(that.p)), f_(boost::move(that.f_)) {}

    void operator()() {
      try {
        p.set_value(f_());
      } catch (...) {}
    }
  };

  task ts(boost::move(f));
  boost::future<result_type> r(ts.p.get_future());
  boost::thread(boost::move(ts));
  return r;
}

int a() {
  std::cout << boost::this_thread::get_id() << '\n';
  return 1;
}

auto main() -> decltype(0) {
  auto r = spawn_task(a);
  r.wait();
  assert(r.is_ready());
  return 0;
}
