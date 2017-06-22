#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>

template <typename T>
T do_some() {
  std::cout << std::this_thread::get_id() << '\n';
}

template <typename T>
T doit() {
  {
    std::packaged_task<T()> t;
  }

  {
    std::packaged_task<T()> t(do_some<T>);
    std::future<T> f = t.get_future();
    std::thread th(std::move(t));
    th.detach();
    std::cout << f.get() << '\n';
  };

  {
    std::packaged_task<T()> t {
      std::allocator_arg,
      std::allocator<std::packaged_task<int()> >(),
      do_some<T>
    };
    std::future<T> f = t.get_future();
    std::thread th(std::move(t));
    th.detach();
    std::cout << f.get() << '\n';
  }

  {
    std::packaged_task<T()> t1(do_some<T>);
    std::future<T> f1 = t1.get_future();
    std::packaged_task<T()> t2 = std::move(t1);
//    std::future<T> f2 = t2.get_future();
    std::thread th(std::move(t2));
    th.detach();
    std::cout << f1.get() << '\n';
  };
}

auto main() -> int
{
  doit<int>();
  return 0;
}
