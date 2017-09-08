#include <boost/ref.hpp>
#include <boost/utility.hpp>

#include <cassert>
#include <string>
#include <vector>

#include "thread_only.hpp"

// Test1
bool normal_function_called = false;

void normal_function() {
  normal_function_called = true;
}

int nfoa_res = 0;

void normal_function_one_arg(int i) {
  nfoa_res = i;
}

// Test2
struct callable_no_args {
  static bool called;
  void operator()() const {
    called = true;
  }
};

bool callable_no_args::called = false;

// Test3
struct callable_noncopyable_no_args : boost::noncopyable {
  callable_noncopyable_no_args() : boost::noncopyable() {}
  static bool called;
  void operator()() {
    called = true;
  }
};

bool callable_noncopyable_no_args::called = false;

// Test4
struct callable_one_arg {
  static bool called;
  static int called_arg;
  void operator()(int arg) const {
    called = true;
    called_arg = arg;
  }
};

bool callable_one_arg::called = false;
int callable_one_arg::called_arg = 0;

// Test5
struct callable_multiple_arg {
  static bool called_two;
  static int called_two_arg1;
  static double called_two_arg2;
  static bool called_three;
  static std::string called_three_arg1;
  static std::vector<int> called_three_arg2;
  static int called_three_arg3;

  void operator()(int arg1, double arg2) const {
    called_two = true;
    called_two_arg1 = arg1;
    called_two_arg2 = arg2;
  }

  void operator()(std::string const& arg1, std::vector<int> const& arg2,
    int arg3) const {
    called_three = true;
    called_three_arg1 = arg1;
    called_three_arg2 = arg2;
    called_three_arg3 = arg3;
  }
};

bool callable_multiple_arg::called_two = false;
int callable_multiple_arg::called_two_arg1;
double callable_multiple_arg::called_two_arg2;
bool callable_multiple_arg::called_three = false;
std::string callable_multiple_arg::called_three_arg1;
std::vector<int> callable_multiple_arg::called_three_arg2;
int callable_multiple_arg::called_three_arg3;

void test_1() {
  boost::thread t(normal_function);
  t.join();
  assert(normal_function_called);
}

void test_2() {
  callable_no_args f;
  boost::thread t(f);
  t.join();
  assert(f.called);
}

void test_3() {
  callable_noncopyable_no_args f;
  boost::thread t(boost::ref(f));
  t.join();
  assert(callable_noncopyable_no_args::called);
}

void test_4() {
  callable_one_arg f;
  boost::thread t(f, 42);
  t.join();
  assert(callable_one_arg::called);
  assert(callable_one_arg::called_arg == 42);
}

void test_5() {
  std::vector<int> x;
  for (unsigned i = 0; i < 10; ++i)
    x.push_back(i*i);
  callable_multiple_arg f;
  boost::thread t1(f, "a", x, 1);
  t1.join();
  assert(callable_multiple_arg::called_three);
  assert(callable_multiple_arg::called_three_arg1 == "a");
  assert(callable_multiple_arg::called_three_arg2.size() == x.size());
  assert(callable_multiple_arg::called_three_arg3 == 1);

  for (unsigned j = 0; j < x.size(); ++j)
    assert(callable_multiple_arg::called_three_arg2.at(j) == x[j]);

  double const d1 = 1.234;
  boost::thread t2(f, 2, d1);
  t2.join();
  assert(callable_multiple_arg::called_two);
  assert(callable_multiple_arg::called_two_arg1 == 2);
  assert(callable_multiple_arg::called_two_arg2 == 1.234);
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4(); test_5();
  return 0;
}
