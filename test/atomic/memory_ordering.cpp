#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/time_formatters.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/barrier.hpp>

#include "../macro/config.hpp"

template <boost::memory_order store, boost::memory_order load>
class test_total_store_order {
public:
  test_total_store_order(void);
  void run(boost::posix_time::time_duration& timeout);
  bool is_conflict(void) const { return is_conflict_; }
private:
  void thread1(void);
  void thread2(void);
  void check_conflict(void);

  boost::atomic<int> a_;

  // Insert a bit of padding to push the two variables into different cache li-
  // nes and increase the likelihood of detecting a conflict
  char p1_[512];
  boost::atomic<int> b_;

  char p2_[512];
  boost::barrier barrier_;

  int verify_b1_, verify_a2_;

  boost::atomic<bool> terminate_threads_;
  boost::atomic<int> termination_consensus_;

  bool is_conflict_;
  boost::mutex m_;
  boost::condition_variable c_;
};

template <boost::memory_order store, boost::memory_order load>
test_total_store_order<store, load>::test_total_store_order(void)
  : a_(0), b_(0), barrier_(2) {}

template <boost::memory_order store, boost::memory_order load>
void test_total_store_order<store, load>::run(
  boost::posix_time::time_duration& timeout) {
  boost::system_time start = boost::get_system_time();
  boost::system_time end = start + timeout;

  boost::thread t1(boost::bind(&test_total_store_order::thread1, this));
  boost::thread t2(boost::bind(&test_total_store_order::thread2, this));

  {
    boost::mutex::scoped_lock l(m_);
    while (boost::get_system_time() < end && !is_conflict_)
      c_.timed_wait(l, end);
  }

  terminate_threads_.store(true, boost::memory_order_relaxed);

  t2.join();
  t1.join();

  boost::posix_time::time_duration duration = boost::get_system_time() - start;

  if (duration < timeout)
    timeout = duration;
}

volatile int backoff_dummy;

template <boost::memory_order store, boost::memory_order load>
void test_total_store_order<store, load>::thread1(void) {
  for (;;) {
    a_.store(1, store);
    int b = b_.load(load);

    barrier_.wait();

    verify_b1_ = b;

    barrier_.wait();

    check_conflict();

    // Both threads synchronize via barriers. So either both threads must exit
    // here. Or they must both do another round. Otherwise one of them will wait
    // forever.
    if (terminate_threads_.load(boost::memory_order_relaxed))
      for (;;) {
        int tmp =
          termination_consensus_.fetch_or(1, boost::memory_order_relaxed);

        if (tmp == 3) return;
        if (tmp & 4) break;
      }
      termination_consensus_.fetch_xor(4, boost::memory_order_relaxed);

      unsigned int delay = rand() % 10000;
      a_.store(0, boost::memory_order_relaxed);

      barrier_.wait();

      while (--delay)
        backoff_dummy = delay;
  }
}

template <boost::memory_order store, boost::memory_order load>
void test_total_store_order<store, load>::thread2(void) {
  for (;;) {
    b_.store(1, store);
    int a = a_.load(load);

    barrier_.wait();

    verify_a2_ = a;

    barrier_.wait();

    check_conflict();

    // Both threads synchronize via barriers. So either both threads must exit 
    // here. Or they must both do another round. Otherwise one of them will wait
    // forever.
    if (terminate_threads_.load(boost::memory_order_relaxed))
      for (;;) {
        int tmp =
          termination_consensus_.fetch_or(2, boost::memory_order_relaxed);

      if (tmp == 3) return;
      if (tmp & 4) break;
    }
    termination_consensus_.fetch_xor(4, boost::memory_order_relaxed);

    unsigned int delay = rand() % 10000;
    b_.store(0, boost::memory_order_relaxed);

    barrier_.wait();

    while (--delay)
      backoff_dummy = delay;
  }
}

template <boost::memory_order store, boost::memory_order load>
void test_total_store_order<store, load>::check_conflict(void) {
  if (verify_b1_ == 0 && verify_a2_ == 0) {
    boost::mutex::scoped_lock l(m_);
    is_conflict_ = true;
    terminate_threads_.store(true, boost::memory_order_relaxed);
    c_.notify_all();
  }
}

void test_1(void) {
LOG;
  double sum = 0.0;

  for (unsigned n = 0; n < 10; ++n) {
LOG;
    boost::posix_time::time_duration timeout(0, 0, 10);

    test_total_store_order<boost::memory_order_relaxed,
      boost::memory_order_relaxed> test1;

    test1.run(timeout);

    if (!test1.is_conflict()) {
LOG;
      std::cout 
        << "Failed to detect order=seq_cst violation while Ith order=relaxed \n"
        << "\t-- intrinsic ordering too strong for this test.\n";
      return;
    }
LOG;
    std::cout << "seq_cst viilation with order=relaxed after "
      << boost::posix_time::to_simple_string(timeout) << "\n";

    sum = sum + timeout.total_microseconds();
  }
LOG;
  // Determine maximum likelihood estimate for average time between race observ-
  // ations.
  double avg_race_time = (sum / 10);
  double avg_race_time_confidence = avg_race_time * 2 * 10 / 7.44;
  boost::posix_time::time_duration timeout =
    boost::posix_time::microseconds((long)(5.298 * avg_race_time_confidence));

  std::cout << "Run seq_cst for "
    << boost::posix_time::to_simple_string(timeout) << '\n';

  test_total_store_order<boost::memory_order_seq_cst,
    boost::memory_order_seq_cst> test2;
  test2.run(timeout);

  assert(!test2.is_conflict());
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
