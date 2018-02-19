#include <boost/thread.hpp>

#include <iostream>

struct Callback {
  typedef void result_type;
  result_type operator()() {
    std::cout << "a\n";
  }
};

template <typename T>
typename boost::decay<T>::type decay_copy(T&& v) {
  return boost::forward<T>(v);
}

template <typename T>
typename boost::decay<T>::type doit() {
  return boost::forward<T>(T());
}

auto main() -> decltype(0) {
  auto r = doit<Callback>();
  r();
  return 0;
}
