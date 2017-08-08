#include <utility>

#include <cassert>

#include "thread_only.hpp"

void get_thread_id(boost::thread::id* id) {
  *id = boost::this_thread::get_id();
}

boost::thread make_thread(boost::thread::id* id) {
  return boost::thread(get_thread_id, id);
}

void test_1() {
  boost::thread::id id;
  boost::thread t = boost::thread(get_thread_id, &id);
  boost::thread::id r_id = t.get_id();
  t.join();
  assert(id == r_id);
}

void test_2() {
  boost::thread::id this_id;
  boost::thread t1(get_thread_id, &this_id);
  boost::thread t2;
  t2 = std::move(t1);
  boost::thread::id that_id = t2.get_id();
  t2.join();
  assert(this_id == that_id);
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}
