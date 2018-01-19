#include <functional>
#include <iostream>
#include <memory>

template <typename T>
using callbacker = std::function<void(T)>;

typedef std::shared_ptr<int> value_type;

void f0(value_type input, callbacker<value_type> cb) {
  *input = 2;
  cb(input);
}

void f1(value_type input, callbacker<value_type> cb) {
  *input = 3;
  cb(input);
}

void f2(value_type input, callbacker<value_type> cb) {
  *input = 3;
  cb(input);
}

void f3(value_type input, callbacker<value_type> cb) {
  *input = 4;
  cb(input);
}

void f4(value_type input, callbacker<value_type> cb) {
  *input = 5;
  cb(input);
}

auto main() -> decltype(0) {
  auto r = std::make_shared<int>(2);
  f0(r, [=](value_type input) {
    *r *= *input;
    f1(r, [=](value_type input) {
      *r *= *input;
      f2(r, [=](value_type input) {
        *r *= *input;
        f3(r, [=](value_type input) {
          *r *= *input;
          f4(r, [=](value_type input) {
            *r *= *input;
          });
        });
      });
    });
  });
  std::cout << *r << '\n';
  return 0;
}
