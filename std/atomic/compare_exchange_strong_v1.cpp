#include <atomic>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

template <typename T>
void doit() {
  std::atomic<T> x(3);
  T expected = 3;
  T desired = 1;
  bool r = std::atomic_compare_exchange_strong(&x, &expected, desired);
  // x == expected, so will be replaced with desired
  // x becomes 1
  std::cout << std::boolalpha << r << " " 
    << x.load() << " " << expected << '\n';

  expected = 2;
  
  // x != expected, so will not be replaced with desired
  // x stays with 1
  r = std::atomic_compare_exchange_strong(&x, &expected, desired);
  std::cout << std::boolalpha << r << " "
    << x.load() << " " << expected << '\n';

  std::thread t([&x, &desired]{
    // x != expected still, so will not be replaced with desired
    // x still stays with 1
    T expected = 2;
    bool r = std::atomic_compare_exchange_strong(&x, &expected, desired);
    std::cout << std::boolalpha << r << " "
      << x.load() << " " << expected << '\n';
  });

  t.join();

  std::vector<std::thread> vt;
  for (T i=0; i<10; ++i)
    vt.push_back(std::thread(
      [&x, &desired, &i] {
        T expected = 1;
        bool r = std::atomic_compare_exchange_strong(&x, &expected, desired);
        std::cout << std::boolalpha << r << " "
          << x.load() << " " << expected << '\n'; 
      })
    );

  for (std::vector<std::thread>::iterator it=vt.begin(); it!=vt.end(); ++it)
    it->join(); 

}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
