#include <iostream>
#include <thread>
#include <vector>

void f1(std::thread t) {
  t.join();
  std::cout << __func__ << ": " << std::this_thread::get_id() << '\n';
}

template <typename T>
T f2() {
  std::vector<int> v;
  std::thread t([&v](int x) {
      for (T i = 0; i < x; ++i)
        v.push_back(i);
    }, 10
  );

  f1(std::move(t));
}

template <typename T>
T doit() {
  f2<T>();
}

auto main() -> int
{
  doit<int>();
  return 0;
}
