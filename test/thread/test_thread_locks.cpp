#include <boost/thread.hpp>

#include <iostream>

#include "../mutex/hpp/shared_mutex.hpp"

class thread_locks {
public:
  shared_mutex m;

  static int function_1(thread_locks *ptr_thread_locks);
  static int function_2(thread_locks *ptr_thread_locks,
    boost::upgrade_lock<shared_mutex>& lock);

  thread_locks() : m() {}
};

int thread_locks::function_1(thread_locks *ptr_thread_locks) {
  std::cout << "Entering " << boost::this_thread::get_id() << " " <<
    "function_1" << '\n';

  boost::upgrade_lock<shared_mutex> lock(ptr_thread_locks->m);

  ptr_thread_locks->function_2(ptr_thread_locks, lock);

  std::cout << "Returned from call " << boost::this_thread::get_id() <<
    " function_1\n";

  return 0;
}

int thread_locks::function_2(thread_locks *,
  boost::upgrade_lock<shared_mutex>& lock) {
  std::cout << "Before exclusive locking " << boost::this_thread::get_id() <<
    " function_2\n";

  return 0;
}

auto main() -> decltype(0) {
  thread_locks locks;
  boost::thread_group threads;

  threads.create_thread(boost::bind(thread_locks::function_1, &locks));
  threads.create_thread(boost::bind(thread_locks::function_1, &locks));
  threads.create_thread(boost::bind(thread_locks::function_1, &locks));
  threads.join_all();

  return 0;
}
