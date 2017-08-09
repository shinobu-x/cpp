#include "utils.hpp"

inline boost::xtime delay(int s, int ms = 0, int ns = 0) {
  const int MILLISECONDS_PER_SECOND = 1000;
  const int NANOSECONDS_PER_SECOND = 1000000000;
  const int NANOSECONDS_PER_MILLISECOND = 1000000;

  boost::xtime xt;

//  if (boost::TIME_UTC_ != boost::xtime_get (&xt, boost::TIME_UTC_))
//    ERROR("boost::timeout_get != boost::TIME_UTC_");

  ns += xt.nsec;
  ms += ns / NANOSECONDS_PER_MILLISECOND;
  s += ms / MILLISECONDS_PER_SECOND;
  ns += (ms % MILLISECONDS_PER_SECOND) * NANOSECONDS_PER_MILLISECOND;
  xt.nsec = ns % NANOSECONDS_PER_SECOND;
  xt.sec += s + (ns / NANOSECONDS_PER_SECOND);

  return xt;
}

inline bool in_range(const boost::xtime& xt, int s=1) {
  boost::xtime min = delay(-s);
  boost::xtime max = delay(0);
  return (boost::xtime_cmp(xt, min) >= 0) &&
    (boost::xtime_cmp(xt, max) <= 0);
}

execution_monitor::execution_monitor(wait_type type, int sec)
  : done_(false), type_(type), sec_(sec) {}

void execution_monitor::start() {
  if (type_ != use_sleep_only)
    boost::unique_lock<boost::mutex> l(m_);
  done_ = false;
}

void execution_monitor::finish() {
  if (type_ != use_sleep_only)
    boost::unique_lock<boost::mutex> l(m_);
    if (type_ == use_condition)
      cond_.notify_one();
  done_ = true;
}

bool execution_monitor::wait() {
  boost::xtime xt = delay(sec_);

  if (type_ != use_condition)
    boost::thread::sleep(xt);

  if (type_ != use_sleep_only) {
    boost::unique_lock<boost::mutex> l(m_);
    while (type_ == use_condition && !done_)
      if (!cond_.timed_wait(l, xt))
        break;
    return done_;
  } 
  return done_;
}

namespace thread_detail_anon {
template<typename R, typename T>
thread_member_binder<R, T>::thread_member_binder(T::*func(), T& param) : func_(func), param_(param) {}
};

#define DEFAULT_EXECUTION_MONITOR_TYPE execution_monitor::use_condition

template <typename F>
void timed_out(F func, int sec,
  execution_monitor::wait_type type = DEFAULT_EXECUTION_MONITOR_TYPE) {
  execution_monitor monitor(type, sec);
//  thread_detail_anon::indirect_adapter<F> ifunc(func, monitor);
//  monitor.start();
//  boost::thread t(ifunc);
//  BOOST_REQUIRE_MESSAGE(monitor.wait(),
//    "Timed test didn't complete in time, passible deadlock.");
}
