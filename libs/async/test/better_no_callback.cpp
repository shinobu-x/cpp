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
      [f, that]() mutable -> decltype(std::declval<F&>()()) {
        that.wait();
        return f();
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

auto main() -> decltype(0) {
  return 0;
}
