#include <functional>
#include <iostream>
#include <memory>

template <typename T>
using callback = std::function<void(T)>;

void f0(std::shared_ptr<int> input, std::function<void(void)> f) { f(); }

void f1(std::shared_ptr<int> input, callback<std::shared_ptr<int> > c) {
  std::shared_ptr<int> v(new int(9));
  c(v);
}

void f2(std::shared_ptr<int> input, callback<std::shared_ptr<int> > c) {
  std::shared_ptr<int> v(new int(8));
  c(v);
}

void f3(std::shared_ptr<int> input, callback<std::shared_ptr<int> > c) {
  std::shared_ptr<int> v(new int(7));
  c(v);
}

void f4(std::shared_ptr<int> input, callback<std::shared_ptr<int> > c) {
  std::shared_ptr<int> v(new int(6));
  c(v);
}

auto main() -> decltype(0) {
  std::shared_ptr<int> n(new int(2));
  auto r = std::make_shared<int>(2);

  f0(n, [=](){
    std::shared_ptr<int> n1(new int(3));
    *r = (*n1)*(*n);
    f1(r, [=](std::shared_ptr<int> input) {
      std::shared_ptr<int> n2(new int(4));
      *r = (*n2)*(*input);
      f2(r, [=](std::shared_ptr<int> input) {
        std::shared_ptr<int> n3(new int(5));
        *r = (*n3)*(*input);
        f3(r, [=](std::shared_ptr<int> input) {
          std::shared_ptr<int> n4(new int(6));
          *r = (*n4)*(*input);
          f4(r, [=](std::shared_ptr<int> input) {
            std::shared_ptr<int> n5(new int(7));
            *r = (*n5)*(*input);
          });
        });
      });
    });
  });

  std::cout << *r << '\n';

  return 0;
}
