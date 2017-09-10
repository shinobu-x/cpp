#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include "../../mutex/hpp/shared_mutex.hpp"

template <typename lock_type>
class locking_thread {
  shared_mutex& rw_mutex_;
  unsigned& unblocked_count_;
  boost::condition_variable& unblocked_condition_;
  unsigned& simultaneous_running_count_;
  unsigned& max_simultaneous_running_;
  boost::mutex& unblocked_count_mutex_;
  boost::mutex& finish_mutex_;
public:
  locking_thread(shared_mutex&, unsigned&, boost::mutex&, 
    boost::condition_variable&, boost::mutex&, unsigned&, unsigned&);

  void operator()();

private:
  void operator=(locking_thread&);
};

#pragma once
#include "../ipp/shared_mutex_locking_thread.ipp"
