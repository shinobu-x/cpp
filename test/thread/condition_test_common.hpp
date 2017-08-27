#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread_time.hpp>

unsigned const timeout_seconds = 5;

struct wait_for_flag {
  boost::mutex m_;
  boost::condition_variable cond_;
  bool flag_;
  unsigned woken_;

  wait_for_flag()
    : flag_(false), woken_(0) {}
/*
  wait_for_flag(const wait_for_flag&) = delete;
  wait_for_flag(wait_for_flag&&) = delete;
  wait_for_flag& operator=(const wait_for_flag&) = delete;
  wait_for_flag& operator=(wait_for_flag&&) = delete;
*/
  struct check_flag {
    check_flag(bool const& flag)
      : flag_(flag) {}

    bool operator()() const {
      return flag_;
    }
  private:
    bool const& flag_;
    void operator=(check_flag&);
  };

  void wait_without_predicate() {
    boost::unique_lock<boost::mutex> l(m_);

    while (!flag_)
      cond_.wait(l);

    ++woken_;
  }

  void wait_with_predicate() {
    boost::unique_lock<boost::mutex> l(m_);
    cond_.wait(l, check_flag(flag_));

    if (flag_)
      ++woken_;
  }

  void timed_wait_without_predicate() {
    boost::system_time const timeout =
      boost::get_system_time() + boost::posix_time::seconds(timeout_seconds);
    boost::unique_lock<boost::mutex> l(m_);

    while (!flag_)
      if (!cond_.timed_wait(l, timeout))
        return;

    ++woken_;
  }

  void timed_wait_with_predicate() {
    boost::system_time const timeout =
      boost::get_system_time() + boost::posix_time::seconds(timeout_seconds);
    boost::unique_lock<boost::mutex> l(m_);

    if (cond_.timed_wait(l, timeout, check_flag(flag_)) && flag_)
      ++woken_;

  }

  void relative_timed_wait_with_predicate() {
    boost::unique_lock<boost::mutex> l(m_);

    if (cond_.timed_wait(l, boost::posix_time::seconds(timeout_seconds),
      check_flag(flag_)) && flag_)
      ++woken_;
  }

};

