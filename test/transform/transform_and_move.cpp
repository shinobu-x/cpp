#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

template <typename input, typename callable,
  typename output = typename std::result_of<callable(input)>::type>
std::vector<output> transform_and_copy(const std::vector<input>& in,
  callable c) {
  typedef std::vector<output> value_type;
  value_type out(in.size());
  std::transform(in.cbegin(), in.cend(), out.begin(), c);

  return out;
}

void doit() {
  std::vector<int> v;
  for (int i = 0; i < 10; ++i) {
    v.push_back(i);
  }

  for (auto e : transform_and_copy(v, [](int i) { return i * i; })) {
    std::cout << e;
    e != 81 ? std::cout << ", " : std::cout << '\n';
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
