#include <boost/asio/io_service.hpp>

#include <iostream>
#include <thread>
#include <utility>
#include <vector>

void test_1() {
  std::cout << __func__ << '\n';
  boost::asio::io_service ios;
  ios.post([&ios](){std::cout << 0 << '\n';});
  ios.post([&ios](){std::cout << 1 << '\n';});
  ios.post([&ios](){std::cout << 2 << '\n';});
  ios.post([&ios](){std::cout << 3 << '\n';});
  ios.post([&ios](){std::cout << 4 << '\n';});
  ios.run();
}

void test_2() {
  std::cout << __func__ << '\n';
  boost::asio::io_service ios;
  ios.post([&ios] {
    std::cout << 0 << '\n';
    ios.post([&ios] {
      std::cout << 1 << '\n';
      ios.post([&ios] {
        std::cout << 2 << '\n';
        ios.post([&ios] {
          std::cout << 3 << '\n';
          ios.post([&ios] {
            std::cout << 4 << '\n';
          });
        });
      });
    });
  });
  ios.run();
}
void test_3() {
  std::cout << __func__ << '\n';
  boost::asio::io_service ios;
  std::vector<std::thread> v;
  ios.post([&ios](){std::cout << 0 << '\n';});
  ios.post([&ios](){std::cout << 1 << '\n';});
  ios.post([&ios](){std::cout << 2 << '\n';});
  ios.post([&ios](){std::cout << 3 << '\n';});
  ios.post([&ios](){std::cout << 4 << '\n';});
  v.emplace_back([&ios]{ios.run();});
  v.emplace_back([&ios]{ios.run();});
  v.emplace_back([&ios]{ios.run();});
  v.emplace_back([&ios]{ios.run();});
  v.emplace_back([&ios]{ios.run();});
  for (auto& i : v)
    i.join();
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3();
  return 0;
}
