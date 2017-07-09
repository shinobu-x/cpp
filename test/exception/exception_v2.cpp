#include <exception>
#include <iostream>
#include <thread>

auto main() -> decltype(0)
{
  std::exception_ptr ep;

  try {
    std::thread t([&ep]{
      try {
        throw std::exception();
      } catch (...) {
        std::cout << "Exception" << std::this_thread::get_id() << '\n';
        ep = std::current_exception();
      }
    });

    t.join();

    if (ep)
      std::rethrow_exception(ep);

  } catch (...) {
    std::cout << "Exception" << '\n';
  }
}
