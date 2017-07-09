#include <exception>
#include <iostream>
#include <thread>

auto main() -> decltype(0)
{
  try {
    std::thread t([](){
      try {
        throw std::exception();
      } catch (...) {
        std::cout << "Exception" << std::this_thread::get_id() << '\n';
        throw;
      }
    });

    t.join();

  } catch (...) {
    std::cout << "Exception" << '\n';
  }
}
