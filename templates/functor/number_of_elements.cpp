#include <iostream>
#include <map>
#include <vector>

template <typename T, typename S>
struct get_size {
  S operator()(const T& x) {
    return x.size();
  }
  get_size(int){}
};

struct get_one {
  template <typename T>
  size_t operator()(const T&) const {
    return 1;
  }
  get_one(int) {}
};

template <typename T>
get_size<T, typename T::size_type> test(const T* x) {
  return 0;
}

get_one test(const void*) {
  return 0;
}

template <typename T>
size_t num_of_elem(const T& x) {
  return test(&x)(x);
}

template <typename T>
T doit() {
  std::vector<T> v;
  std::map<T, std::string> m;
  T a;

  for (T i=0; i<10; ++i) {
    v.push_back(i);
    m[i] = "a";
  }

  std::cout << num_of_elem(v) << '\n';
  std::cout << num_of_elem(m) << '\n';
  std::cout << num_of_elem(a) << '\n';
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
