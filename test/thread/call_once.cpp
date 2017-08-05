#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/once.hpp>

#include <cassert>

#include "../macro/config.hpp"

boost::once_flag f = BOOST_ONCE_INIT;
int var_to_init = 0;
boost::mutex m;

void initialize_variable() {
  boost::unique_lock<boost::mutex> l(m);
  ++var_to_init;
}

void call_once_thread() {
  unsigned const loop_count = 100;
  int once_value = 0;

  for (unsigned i = 0; i < loop_count; ++i) {
    boost::call_once(f, initialize_variable);
    once_value = var_to_init;

    if (once_value != 1) break;
  }

  boost::unique_lock<boost::mutex> l(m);
  assert(once_value == 1);
}

void test_1() {
LOG;

  unsigned const num_threads = 20;
  boost::thread_group tg;

  try {
    for (unsigned i = 0; i < num_threads; ++i)
      tg.create_thread(&call_once_thread);
    tg.join_all();
  } catch (...) {
    tg.interrupt_all();
    tg.join_all();
    throw;
  }

  assert(var_to_init == 1);
}

int var_to_init_with_functor = 0;

struct increment_value {
public:
  explicit increment_value(int* value) : value_(value) {}

  void operator() () const {
    boost::unique_lock<boost::mutex> l(m);
    ++(*value_);
  }
private:
  int* value_;
};

void call_once_with_functor() {
  unsigned const loop_count = 100;
  int once_value = 0;
  static boost::once_flag functor_flag = BOOST_ONCE_INIT;

  for (unsigned i = 0; i < loop_count; ++i) {
    boost::call_once(functor_flag, increment_value(&var_to_init_with_functor));
    once_value = var_to_init_with_functor;

    if (once_value != 1) break;

  }

  boost::unique_lock<boost::mutex> l(m);
  assert(once_value == 1);
}

void test_2() {
LOG;

  unsigned const num_threads = 20;
  boost::thread_group tg;

  try {
    for (unsigned i = 0; i < num_threads; ++i)
      tg.create_thread(&call_once_with_functor);
    tg.join_all();
  } catch (...) {
    tg.interrupt_all();
    tg.join_all();
    throw;
  }

  assert(var_to_init_with_functor == 1);
}

struct throw_before_third_pass {
  struct my_exception {};

  static unsigned pass_counter;

  void operator() () const {
    boost::unique_lock<boost::mutex> l(m);
    ++pass_counter;

    if(pass_counter < 3) throw my_exception();
  }
};

unsigned throw_before_third_pass::pass_counter = 0;
unsigned exception_counter = 0;

void call_once_with_exception() {
  static boost::once_flag functor_flag = BOOST_ONCE_INIT;

  try {
    boost::call_once(functor_flag, throw_before_third_pass());
  } catch (throw_before_third_pass::my_exception) {
    boost::unique_lock<boost::mutex> l(m);
    ++exception_counter;
  }
}

void test_3() {
LOG;

  unsigned const num_threads = 20;
  boost::thread_group tg;

  try {
    for (unsigned i = 0; i < num_threads; ++i)
      tg.create_thread(&call_once_with_exception);
    tg.join_all();
  } catch (...) {
    tg.interrupt_all();
    tg.join_all();
    throw;
  }

  assert(throw_before_third_pass::pass_counter == 3u);
  assert(exception_counter == 2u);
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3();
  return 0;
}
