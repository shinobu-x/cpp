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
  void operator=(locking_thread&);
public:
  locking_thread(
    shared_mutex&, 
    unsigned&, boost::mutex&, 
    boost::condition_variable&,
    boost::mutex&,
    unsigned&, unsigned&);
  void operator()();
};

class simple_writing_thread {
  shared_mutex& rwm_mutex_;
  boost::mutex& finish_mutex_;
  boost::mutex& unblocked_mutex_;
  unsigned& unblocked_count_;
  void operator=(simple_writing_thread&);
public:
  simple_writing_thread(
    shared_mutex&,
    boost::mutex&,
    boost::mutex&,
    unsigned&);
  void operator()();
};

class simple_reading_thread {
  shared_mutex& rwm_mutex_;
  boost::mutex& finish_mutex_;
  boost::mutex& unblocked_mutex_;
  unsigned& unblocked_count_;
  void operator=(simple_reading_thread&);
public:
  simple_reading_thread(
    shared_mutex&,
    boost::mutex&,
    boost::mutex&,
    unsigned&);
  void operator()();
};

class simple_upgrade_thread {
  shared_mutex& rwm_mutex_;
  boost::mutex& finish_mutex_;
  boost::mutex& unblocked_mutex_;
  unsigned& unblocked_count_;
  void operator=(simple_upgrade_thread&);
public:
  simple_upgrade_thread(
    shared_mutex&,
    boost::mutex&,
    boost::mutex&,
    unsigned&);
  void operator()();
};

#pragma once
#include "../ipp/shared_mutex_locking_thread.ipp"
