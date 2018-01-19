#include <future>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

template <typename T>
struct Future : std::shared_future<T> {
  using std::shared_future<T>::shared_future;

  template <typename F>
  auto operator->*(F f)&&->Future<
    std::decay_t<decltype(f(std::declval<T const&>()))> > {
    auto that = std::move(*this);
    return std::async(std::launch::async,
      [f, that]() mutable -> decltype(std::declval<F&>()(
        std::declval<T const&>())) {
        return f(that.get());
      }
    );
  }
};

template <>
struct Future<void> : std::shared_future<void> {
  using std::shared_future<void>::shared_future;

  template <typename F>
  auto operator->*(F f)&&->Future<
    std::decay_t<decltype(f())> > {
    auto that = std::move(*this);
    return std::async(std::launch::async,
      [f, that]() mutable -> decltype(std::declval<F&>()()) {
        that.wait();
        return f();
      }
    );
  }
};

std::shared_ptr<int> f0(std::shared_ptr<int> input) {
  std::shared_ptr<int> v(new int(9));
  *v *= *input;
  return v; 
}

std::shared_ptr<int> f1(std::shared_ptr<int> input) {
  std::shared_ptr<int> v(new int(8));
  *v *= *input;
  return v;
}

std::shared_ptr<int> f2(std::shared_ptr<int> input) {
  std::shared_ptr<int> v(new int(7));
  *v *= *input;
  return v;
}

std::shared_ptr<int> f3(std::shared_ptr<int> input) {
  std::shared_ptr<int> v(new int(6));
  *v *= *input;
  return v;
}

std::shared_ptr<int> f4(std::shared_ptr<int> input) {
  std::shared_ptr<int> v(new int(5));
  *v *= *input;
  return v;
}

auto main() -> decltype(0) {
  auto r = std::make_shared<int>(2);
  Future<std::shared_ptr<int> > f = std::async(std::launch::async, f0, r);
  auto x = std::move(f)->*[=](std::shared_ptr<int> input) {
    return f1(input);
  }->*[=](std::shared_ptr<int> input) {
    return f2(input);
  }->*[=](std::shared_ptr<int> input) {
    return f3(input);
  }->*[=](std::shared_ptr<int> input) {
    return f4(input);
  };

  std::cout << *x.get() << '\n';

  return 0;
}
