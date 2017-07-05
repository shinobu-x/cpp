#include <atomic>
#include <thread>
#include <iostream>

int data;
std::atomic<bool> is_ready(false);

void other() {
  ++data;
  is_ready.store(true, std::memory_order_release);
}

template <typename T>
void doit() {
  T t1([]{
      while (!is_ready.load(std::memory_order_acquire)) {}
      std::cout << data << '\n';
    }
  ); 
  T t2(other);

  t2.join();
  t1.join();
}

auto main() -> decltype(0)
{
  doit<std::thread>();
  return 0;
}
