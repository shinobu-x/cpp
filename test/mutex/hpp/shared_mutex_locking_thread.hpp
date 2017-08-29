#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>

#include "shared_mutex.hpp"

template <typename lock_t>
class locking_thread {
  shared_mutex& rw_mutex_;
  unsigned& unblocked_count_;
  boost::condition_variable& unblocked_condition_;
  unsigned& simultaneous_running_count_;
  unsigned& max_simultaneous_running_;
  boost::mutex& unblocked_count_mutex_;
  boost::mutex& finish_mutex_;

public:
  locking_thread(shared_mutex& rw_mutex,
    unsigned& unblocked_count,
    boost::mutex& unblocked_count_mutex,
    boost::condition_variable& unblocked_condition,
    boost::mutex& finish_mutex,
    unsigned& simultaneous_running_count,
    unsigned& max_simultaneous_running) :
      rw_mutex_(rw_mutex),
      unblocked_count_(unblocked_count),
      unblocked_condition_(unblocked_condition),
      simultaneous_running_count_(simultaneous_running_count),
      max_simultaneous_running_(max_simultaneous_running),
      unblocked_count_mutex_(unblocked_count_mutex),
      finish_mutex_(finish_mutex) {}


  void operator()() {
    lock_t l(rw_mutex_);

    {
      boost::unique_lock<boost::mutex> unblocked_lock(unblocked_count_mutex_);
      ++unblocked_count_;
      unblocked_condition_.notify_one();
      ++simultaneous_running_count_;
      if (simultaneous_running_count_ > max_simultaneous_running_)
        max_simultaneous_running_ = simultaneous_running_count_;
    }

    boost::unique_lock<boost::mutex> finish_lock(finish_mutex_);

    {
      boost::unique_lock<boost::mutex> unblocked_lock(unblocked_count_mutex_);
      --simultaneous_running_count_;
    }
  }

private:
  void operator=(locking_thread&);
};

class simple_writing_thread {
  shared_mutex& rwm_mutex_;
  boost::mutex& finish_mutex_;
  boost::mutex& unblocked_mutex_;
  unsigned& unblocked_count_;

  void operator=(simple_writing_thread&);

public:
  simple_writing_thread(shared_mutex& rwm_mutex,
    boost::mutex& finish_mutex,
    boost::mutex& unblocked_mutex,
    unsigned& unblocked_count) :
      rwm_mutex_(rwm_mutex_),
      finish_mutex_(finish_mutex),
      unblocked_mutex_(unblocked_mutex),
      unblocked_count_(unblocked_count) {}

    void operator()() {
      boost::unique_lock<shared_mutex> l(rwm_mutex_);

      {
        boost::unique_lock<boost::mutex> unblocked_lock(unblocked_mutex_);
        ++unblocked_count_;
      }

      boost::unique_lock<boost::mutex> finish_lock(finish_mutex_);
    }
};

class simple_reading_thread {
  shared_mutex& rwm_mutex_;
  boost::mutex& finish_mutex_;
  boost::mutex& unblocked_mutex_;
  unsigned& unblocked_count_;

  void operator=(simple_reading_thread&);

public:
  simple_reading_thread(shared_mutex& rwm_mutex,
    boost::mutex& finish_mutex,
    boost::mutex& unblocked_mutex,
    unsigned& unblocked_count) :
      rwm_mutex_(rwm_mutex),
      finish_mutex_(finish_mutex),
      unblocked_mutex_(unblocked_mutex),
      unblocked_count_(unblocked_count) {}

  void operator()() {
    boost::shared_lock<shared_mutex> l(rwm_mutex_);

    {
      boost::unique_lock<boost::mutex> unblocked_lock(unblocked_mutex_);
      ++unblocked_count_;
    }

    boost::unique_lock<boost::mutex> finish_lock(finish_mutex_);
  }
}; 

class simple_upgrade_thread {
  shared_mutex& rwm_mutex_;
  boost::mutex& finish_mutex_;
  boost::mutex& unblocked_mutex_;
  unsigned& unblocked_count_;

  void operator=(simple_upgrade_thread);

public:
  simple_upgrade_thread(shared_mutex& rwm_mutex, boost::mutex& finish_mutex,
    boost::mutex& unblocked_mutex, unsigned& unblocked_count) :
    rwm_mutex_(rwm_mutex), finish_mutex_(finish_mutex),
    unblocked_mutex_(unblocked_mutex), unblocked_count_(unblocked_count) {}

  void operator()() {
    boost::upgrade_lock<boost::mutex> rwm_lock(rwm_mutex_);

    {
      boost::unique_lock<boost::mutex> unblocked_lock(unblocked_mutex);
      ++unblocked_count_;
    }

    boost::unique_lock<boost::mutex> finish_lock(finish_mutex);
  }
};
