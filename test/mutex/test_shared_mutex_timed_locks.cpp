#include <boost/thread/thread_only.hpp>
#include <boost/thread/xtime.hpp>

#include "./hpp/shared_mutex_locking_thread.hpp"
#include "../utils/utils.hpp"

void test_1() {
  shared_mutex rwm_mutex;
  boost::mutex finish_mutex;
  boost::mutex unblocked_mutex;
  unsigned unblocked_count = 0;
  boost::unique_lock<boost::mutex> finish_lock(finish_mutex);
  boost::thread writer(
    simple_writing_thread(
      rwm_mutex, finish_mutex, unblocked_mutex, unblocked_count));
  boost::thread::sleep(delay(1));
}

auto main() -> decltype(0) {
  return 0;
}
