#include <boost/array.hpp>
#include <boost/cstdint.hpp>
#include <boost/lockfree/detail/atomic.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>

#include "../macro/config.hpp"

inline boost::xtime delay(int, int, int); 
inline bool in_range(const boost::xtime&, int);
inline void error_msg(char const*, char const*, int);

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

#define DEFAULT_EXECUTION_MONITOR_TYPE execution_monitor::use_condition

template <typename F>
void timed_test(F, int, execution_monitor::wait_type =
  DEFAULT_EXECUTION_MONITOR_TYPE);

namespace thread_detail_anon {
  template <typename R, typename T>
  class thread_member_binder {
  public:
    thread_member_binder(R (T::*func)(), T&);
    void operator()() const;
    thread_member_binder(const thread_member_binder&) = delete;
    thread_member_binder(thread_member_binder&&) = delete;
    thread_member_binder& operator()(thread_member_binder&&) = delete;
  private:
    void operator=(thread_member_binder&);
    R (T::*func_)();
    T& param_;
  };

  template <typename F>
  class indirect_adapter {
  public:
    indirect_adapter(F, execution_monitor&);
    void operator()() const;
  private:
    F func_;
    execution_monitor& monitor_;
    void operator=(indirect_adapter&);
  };
} // namespace

namespace {
template <typename int_type>
int_type generate_id() {
  static boost::lockfree::detail::atomic<int_type> generator(0);
  return ++generator;
}

template <typename int_type, std::size_t bucket>
class static_hashed_set {
public:
  int calc_index(int_type const&);
  bool insert(int_type const&);
  bool find(int_type const&);
  bool erase(int_type const& id);
  std::size_t count_nodes(void) const;
private:
  boost::array<std::set<int_type>, bucket> data_;
  mutable boost::array<boost::mutex, bucket> ref_mutex_;
};
} // namespace

#pragma once
#include "utils.ipp"
