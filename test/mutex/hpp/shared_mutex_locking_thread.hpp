#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include "shared_mutex.hpp"

template <typename lock_t>
class locking_thread {
  boost::shared_mutex rw_mutex_;
  unsigned& unblocked_count_;
  boost::condition_variable& unblocked_condition_;
  unsigned& simultaneous_running_count_;
  unsigned& max_simultaneous_running_;
  boost::mutex& unblocked_count_mutex_;
  boost::mutex& finish_mutex_;

public:
  locking_thread(boost::shared_mutex& rw_mutex,
    unsigned& unblocked_count,
    boost::mutex& unblocked_count_mutex,
    boost::condition_variable& unblocked_condition,
    boost::mutex& finish_mutex_,
    unsigned& simultaneous_running_count,
    unsigned& max_simultaneous_running)
    : rw_mutex_(rw_mutex),
      unblocked_count_(unblocked_count),
      unblocked_count_mutex_(unblocked_count_mutex),
      unblocked_condition_(unblocked_condition),
      finish_mutex_(finish_mutex),
      simultaneous_running_count_(simultaneous_running_count),
      max_simultaneous_running_(max_simultaneous_running) {}
