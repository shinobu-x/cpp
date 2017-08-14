#include <boost/thread/condition.hpp>
#include <boost/thread/xtime.hpp>

#include <cassert>

#include "../thread/thread_only.hpp"
#include "../macro/config.hpp"

struct condition_test_data {
  condition_test_data() 
    : notified_(0), woken_(0) {}
  condition_test_data(const condition_test_data&) = delete;
  condition_test_data(condition_test_data&&) = delete;
  condition_test_data& operator=(const condition_test_data&) = delete;
  condition_test_data& operator=(condition_test_data&&) = delete;

  boost::mutex m_;
  boost::condition_variable cond_;
  int notified_;
  int woken_;
};

struct cond_predicate {
  cond_predicate(int& var, int val)
    : var_(var), val_(val) {}
  cond_predicate(const cond_predicate&) = delete;
  cond_predicate(cond_predicate&&) = delete;
  cond_predicate& operator=(const cond_predicate&) = delete;
  cond_predicate& operator=(cond_predicate&&) = delete;

  bool operator()() { return var_ == val_; }

  int& var_;
  int val_;

private:
  void operator=(cond_predicate&);
};

void condition_test_thread(condition_test_data* data) {
  boost::unique_lock<boost::mutex> l(data->m_);
  while(!(data->notified_ > 0))
    data->cond_.wait(l);
  data->woken_++;
}

void condition_test_waits(condition_test_data* data) {
  boost::unique_lock<boost::mutex> l(data->m_);
  
}
auto main() -> decltype(0) {
  return 0;
}
