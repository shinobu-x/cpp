#include <future>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

namespace {

template <typename T>
struct future : std::shared_future<T> {
  using std::shared_future<T>::shared_future;

  template <typename F>
  auto operator->*(F f)&&->future<
    std::decay_t<decltype(f(std::declval<T const&>()))> > {
    auto that = std::move(*this);

    return std::async(
      std::launch::async,
      [f, that]() mutable -> decltype(std::declval<F&>()(
        std::declval<T const&>())) {
          return f(that.get());
      }
    );
  }
}; // future

template <>
struct future<void> : std::shared_future<void> {
  using std::shared_future<void>::shared_future;

  template <typename F>
  auto operator->*(F f)&&->future<std::decay_t<decltype(f())> > {
    auto that = std::move(*this);
    return std::async(
      std::launch::async,
      [f, that]() mutable -> decltype(std::declval<F&>()()) {
        that.wait();
        return f();
      }
    );
  }
}; // future
} // namespace

std::shared_ptr<int> f0(std::shared_ptr<int> input) {
  std::shared_ptr<int> v(new int(2));
  *v *= *input;
  return v;
}

auto f1(std::shared_ptr<int> input) {
  *input *= *input;
  return input;
}

auto main() -> decltype(0) {
  auto r = std::make_shared<int>(2);
  future<std::shared_ptr<int> > f = std::async(std::launch::async, f0, r);
  auto x = std::move(f)->*[=](std::shared_ptr<int> input){
    std::shared_ptr<int> v(new int(10));
    return f1(v);
  }->*[=](std::shared_ptr<int> input /* returned value from f1 */) {
    *r *= *input;
    return r;
  };
  std::cout << *x.get() << '\n';

  return 0;
}
