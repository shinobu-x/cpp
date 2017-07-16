#include <iostream>
#include <mutex>
#include <string>
#include <thread>

thread_local size_t x = 1;
std::mutex m;

void do_inc(const std::string& s) {
  ++x;
  std::lock_guard<std::mutex> l(m);
  std::cout << s << ": " << x << '\n';
}

auto main() -> decltype(0)
{
  std::thread t1(do_inc, "t1"), t2(do_inc, "t2");

  {
    std::lock_guard<std::mutex> l(m);
    std::cout << __func__ << ": " << x << '\n';
  }

  t1.join();
  t2.join();
}
