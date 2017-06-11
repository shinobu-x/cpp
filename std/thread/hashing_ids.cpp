#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

template <typename T, T N>
T doit() {
  std::vector<std::thread> vt;
  for (T i = 0; i < N; ++i)
    vt.emplace_back([]{
      std::this_thread::sleep_for(std::chrono::seconds(1));
    });

  std::hash<std::thread::id> hash;

  for (auto& v : vt) {
    std::cout << "Thread: " << v.get_id() 
      << " / Hash: " << hash(v.get_id()) << '\n';
    v.join();
  }
}

auto main() -> int
{
  doit<int, 8>();
  return 0;
}
