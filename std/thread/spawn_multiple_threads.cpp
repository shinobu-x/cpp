#include <algorithm>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

template <typename T, T N>
T doit() {
  std::vector<std::thread> ts;
  for (T i = 0; i < N; ++i) {
    ts.push_back(std::thread([&i](){
      std::cout << std::this_thread::get_id() << '\n'; }));
  }

  std::for_each(ts.begin(), ts.end(),
    std::mem_fn(&std::thread::join));
}

auto main() -> int {
  doit<int, 20>();
  return 0;
}
