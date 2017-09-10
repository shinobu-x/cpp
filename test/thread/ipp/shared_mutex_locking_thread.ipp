#include "../hpp/shared_mutex_locking_thread.hpp"

template <typename lock_type>
locking_thread<lock_type>::locking_thread(
  shared_mutex& rw_mutex,
  unsigned& unblocked_count,
  boost::mutex& unblocked_count_mutex,
  boost::condition_variable& unblocked_condition,
  boost::mutex& finish_mutex,
  unsigned& simultaneous_running_count,
  unsigned& max_simultaneous_running)
  : rw_mutex_(rw_mutex),
    unblocked_count_(unblocked_count),
    unblocked_condition_(unblocked_condition),
    simultaneous_running_count_(simultaneous_running_count),
    max_simultaneous_running_(max_simultaneous_running),
    unblocked_count_mutex_(unblocked_count_mutex),
    finish_mutex_(finish_mutex) {}

template <typename lock_type>
void locking_thread<lock_type>::operator()() {
  lock_type lock(rw_mutex_);

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex_);
    ++unblocked_count_;
    unblocked_condition_.notify_one();
    ++simultaneous_running_count_;

    if (simultaneous_running_count_ > max_simultaneous_running_)
      max_simultaneous_running_ = simultaneous_running_count_;
  }

  boost::unique_lock<boost::mutex> unblock(finish_mutex_);

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_count_mutex_);
    --simultaneous_running_count_;
  }
}

simple_writing_thread::simple_writing_thread(
  shared_mutex& rwm_mutex,
  boost::mutex& finish_mutex,
  boost::mutex& unblocked_mutex,
  unsigned& unblocked_count)
  : rwm_mutex_(rwm_mutex),
    finish_mutex_(finish_mutex),
    unblocked_mutex_(unblocked_mutex),
    unblocked_count_(unblocked_count) {}

void simple_writing_thread::operator()() {
  boost::unique_lock<shared_mutex> lock(rwm_mutex_);

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_mutex_);
    ++unblocked_count_;
  }

  boost::unique_lock<boost::mutex> finish_lock(finish_mutex_);
}

simple_reading_thread::simple_reading_thread(
  shared_mutex& rwm_mutex,
  boost::mutex& finish_mutex,
  boost::mutex& unblocked_mutex,
  unsigned& unblocked_count)
  : rwm_mutex_(rwm_mutex),
    finish_mutex_(finish_mutex),
    unblocked_mutex_(unblocked_mutex),
    unblocked_count_(unblocked_count) {}

void simple_reading_thread::operator()() {
  boost::shared_lock<shared_mutex> lock(rwm_mutex_);

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_mutex_);
    ++unblocked_count_;
  }

  boost::unique_lock<boost::mutex> finish_lock(finish_mutex_);
}

simple_upgrade_thread::simple_upgrade_thread(
  shared_mutex& rwm_mutex,
  boost::mutex& finish_mutex,
  boost::mutex& unblocked_mutex,
  unsigned& unblocked_count)
  : rwm_mutex_(rwm_mutex),
    finish_mutex_(finish_mutex),
    unblocked_mutex_(unblocked_mutex),
    unblocked_count_(unblocked_count) {}

void simple_upgrade_thread::operator()() {
  boost::upgrade_lock<shared_mutex> lock(rwm_mutex_);

  {
    boost::unique_lock<boost::mutex> unblock(unblocked_mutex_);
    ++unblocked_count_;
  }

  boost::unique_lock<boost::mutex> finish_lock(finish_mutex_);
}
