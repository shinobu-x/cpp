#include <cassert>
#include <future>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

  template <typename T>
  struct future : std::shared_future<T> {
    using std::shared_future<T>::shared_future; // want to use shared_futuere

    template <typename F> // collable
    auto operator->*(F f)&&->future<
      std::decay_t<
        decltype(f(std::declval<T const&>()))
      >
    > {
        auto that = std::move(*this);

        return std::async(
          std::launch::async,
          [f, that]() mutable ->
            decltype(std::declval<F&>()(std::declval<T const&>())) {
            return f(that.get());
          }
        );
      }
  }; // future

  template <>
  struct future<void> : std::shared_future<void> {
    using std::shared_future<void>::shared_future;

    template <typename F>
    auto operator->*(F f)&&->future<
      std::decay_t<
        decltype(f())
      >
    > {
        auto that = std::move(*this);

        return std::async(
          std::launch::async,
          [f, that]() mutable ->
            decltype(std::declval<F&>()()) {
            that.wait();
            return f();
          }
        );
      }
  }; // future
} // namespace

template <typename T>
struct data {
  T _value;
  data(T value) : _value(value) {}
  T get_data() {
    return _value;
  }
  void set_data(T& value) {
    _value = value;
  }
};

auto main() -> decltype(0) {
  std::vector<data<int> > v;
  future<std::vector<data<int> > > f = std::async(std::launch::async,
    [=](std::vector<data<int> > input){ return input; }, v);
  auto x = std::move(f)->*[=](std::vector<data<int>> input) {
    for (int i = 0; i < 100; ++i) {
      data<int> d(i);
      input.push_back(d);
    }
    for (auto& v : input) {
      auto old_value = v.get_data();
      auto new_value = old_value*2;
      v.set_data(new_value);
    }
    return input;
  };
  assert(x.get().size() == 100);

  return 0;
}
