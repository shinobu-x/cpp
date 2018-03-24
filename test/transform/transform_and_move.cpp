#include <algorithm>
#include <deque>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <vector>

template <typename input, typename callable,
  typename output = typename std::result_of<callable(input)>::type>
std::vector<output> transform_and_copy_v1(const std::vector<input>& in,
  callable c) {
  typedef std::vector<output> value_type;
  value_type out(in.size());
  std::transform(in.cbegin(), in.cend(), out.begin(), c);

  return out;
}

template <template <
  typename T, typename A = std::allocator<T> > class container,
  typename input,
  typename callable,
  typename output = typename std::result_of_t<callable(input)> >
container<output> transform_and_copy_v2(
  const container<input>& in, callable c) {
  container<output> out;
  std::transform(in.cbegin(), in.cend(), std::back_inserter(out), c);

  return out;
}

template <typename random_access_iterator, typename callable,
  typename input = typename std::iterator_traits<
    random_access_iterator>::value_type,
  typename output = typename std::result_of_t<callable(input)> >
std::vector<output> transform_and_copy_v3(random_access_iterator begin,
  random_access_iterator end, callable c) {
  std::vector<output> t(end - begin);
  std::transform(begin, end, t.begin(), c);

  return std::move(t);
}

template <typename range, typename callable,
  typename input = typename range::value_type,
  typename output = typename std::result_of_t<callable(input)> >
std::vector<output> transform_and_copy_v4(const range& rng, callable c) {
  std::vector<output> t(rng.size());
  std::transform(rng.begin(), rng.end(), t.begin(), c);

  return std::move(t);
}

void doit() {
  std::vector<int> v;

  for (int i = 0; i < 10; ++i) {
    v.push_back(i);
  }

  for (auto e : transform_and_copy_v1(v, [](int i) { return i * i; })) {
    std::cout << e;
    e != 81 ? std::cout << ", " : std::cout << '\n';
  }
  for (auto e : transform_and_copy_v2(v, [](int i) { return i * i; })) {
    std::cout << e;
    e != 81 ? std::cout << ", " : std::cout << '\n';
  }

  for (auto e : transform_and_copy_v3(v.begin(), v.end(),
    [](int i) { return i * i; })) {
    std::cout << e;
    e != 81 ? std::cout << ", " : std::cout << '\n';
  }

  for (auto e : transform_and_copy_v4(v, [](int i) { return i * i; })) {
    std::cout << e;
    e != 81 ? std::cout << ", " : std::cout << '\n';
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
