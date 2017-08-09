#include <boost/thread/xtime.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>

#include "../macro/config.hpp"

inline boost::xtime delay(int, int, int); 
inline bool in_range(const boost::xtime&, int);

class execution_monitor {
public:
  enum wait_type {
   use_sleep_only,
   use_mutex,
   use_condition
  };

  execution_monitor(wait_type, int);
  execution_monitor(const execution_monitor&) = delete;
  execution_monitor& operator= (const execution_monitor&) = delete;
  execution_monitor(execution_monitor&&) = delete;
  execution_monitor& operator= (execution_monitor&&) = delete;

  void start();
  void finish();
  bool wait();

private:
  boost::mutex m_;
  boost::condition cond_;
  bool done_;
  wait_type type_;
  int sec_;
};

#pragma once
#include "utils.ipp"
