#include <iostream>
#include <thread>

auto main() -> int
{
  std::thread t1([](){
    std::cout << std::this_thread::get_id() << '\n';
    std::thread t2([](){
      std::cout << std::this_thread::get_id() << '\n';
      std::thread t3([](){
        std::cout << std::this_thread::get_id() << '\n';
      });
      t3.join();
    });
    t2.join();
  });
  t1.join();
  return 0;
}
