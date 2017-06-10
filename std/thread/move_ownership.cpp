#include <iostream>
#include <system_error>
#include <thread>
#include <utility>

template <typename T>
T doit() {
  std::thread t1(
    []{ std::cout << std::this_thread::get_id() << '\n'; }
  );
  std::thread t2;

  t2 = std::move(t1);
  t2.join();
}

auto main() -> int
{
  doit<int>();
  return 0;
}
