#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

template <typename T>
void doit() {
  std::mutex m;

  std::thread t([&m]{
    { 
      std::lock_guard<std::mutex> l(m);
      std::cout << "Thread ID: " << std::this_thread::get_id() << "\n"
        << "Pthread ID: " << pthread_self() << '\n';
    }
  });

  {
    std::lock_guard<std::mutex> l(m);
    std::cout << "This thread ID: " << t.get_id() << '\n'
     << "Native handle = " << t.native_handle() << '\n';

    std::this_thread::sleep_for(std::chrono::seconds(3));
  }

  t.join();
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
