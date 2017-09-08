#include <boost/ref.hpp>
#include <boost/utility.hpp>

#include <cassert>
#include <string>
#include <vector>

#include "thread_only.hpp"

bool normal_function_called = false;

void normal_function() {
  normal_function_called = true;
}

int nfoa_res = 0;

void normal_function_one_arg(int i) {
  nfoa_res = i;
}

void test_1() {
  boost::thread t(normal_function_one_arg, 42);
  f.join();
  assert(42 == nfoa_res);
}

struct callable_no_args {
  static bool called;

  void operator()() const {
    called = true;
  }
};

bool callable_no_args::called = false;

void test_2() {
  callable_no_args f;
  boost::thread t(f);
  t.join();
  assert(callable_no_args::called);
}

struct callable_noncopyable_no_args : boost::noncopyable {
  callable_noncopyable_no_args() : boost::noncopyable() {}
  static bool called;

  void operator()() {
    called = true;
  }
};

bool callable_noncopyable_no_args::called = false;

void test_3() {
  callable_noncopyableno_args f;
  boost::thread t(boost::ref(f));
  t.join();
  assert(callable_noncopyable_no_args::called);
}

auto main() -> decltype(0) {
  return 0;
}
