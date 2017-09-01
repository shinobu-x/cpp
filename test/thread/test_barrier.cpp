#include <boost/thread/detail/config.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>

#include <cassert>
#include <vector>

const int number_of_threads = 5;
boost::barrier gen_barrier(number_of_threads);
boost::mutex m;
long global_parameter;

void barrier_thread() {
  for (int i = 0; i < 5; ++i)
    if (gen_barrier.wait()) {
      boost::unique_lock<boost::mutex> l(m);
      global_parameter++;
    }
}

void test_1() {
  boost::thread_group threads;
  global_parameter = 0;

  try {
    for (int i = 0; i < number_of_threads; ++i)
      threads.create_thread(&barrier_thread);
    threads.join_all();
  } catch (...) {
    assert(false);
    threads.interrupt_all();
    threads.join_all();
  }

  assert(global_parameter == 5);
}


auto main() -> decltype(0) {
  test_1();
  return 0;
}
