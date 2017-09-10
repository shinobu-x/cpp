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

inline void error_msg(char const* msg, char const* file, int line) {
  std::cout << "[" << line << "]" << msg << "\n";
}

#define CHECK_MESSAGE(P, M) \
  ((P) ? (void)0 : error_msg((M), __FILE__, __LINE__))

#define REQUIRE_MESSAGE(P, M) \
  CHECK_MESSAGE((P), (M))

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

template <typename R, typename T>
thread_detail_anon::thread_member_binder<R, T>::thread_member_binder(
  R (T::*func)(), T& param)
  : func_(func), param_(param) {}

template <typename R, typename T>
void thread_detail_anon::thread_member_binder<R, T>::operator()() const {
  (param_.*func_)();
}

template <typename F>
thread_detail_anon::indirect_adapter<F>::indirect_adapter(
  F func, execution_monitor& monitor)
  : func_(func), monitor_(monitor) {}

template <typename F>
void thread_detail_anon::indirect_adapter<F>::operator()() const {
  try {
    boost::thread t(func_);
    t.join();
  } catch (...) {
    monitor_.finish();
    throw;
  }
}

template <typename F>
void timed_test(F func, int sec,
  execution_monitor::wait_type type) {
  execution_monitor monitor(type, sec);
  thread_detail_anon::indirect_adapter<F> ifunc(func, monitor);
  monitor.start();
  boost::thread t(ifunc);
  REQUIRE_MESSAGE(monitor.wait(),
    "Timed test didn't complete in time, passible deadlock.");
}

template <typename int_type>
int_type generate_id(void) {
  static boost::lockfree::detail::atomic<int_type> generator(0);
  return ++generator;
}

template <typename int_type, std::size_t bucket>
int static_hashed_set<int_type, bucket>::calc_index(int_type const& id) {
  std::size_t factor =
    std::size_t((float)bucket * 1.616f);
    return ((std::size_t)id * factor) % bucket;
}

template <typename int_type, std::size_t bucket>
bool static_hashed_set<int_type, bucket>::insert(int_type const& id) {
  std::size_t index = calc_index(id);
  boost::lock_guard<boost::mutex> l(static_hashed_set::ref_mutex_[index]);
  std::pair<typename std::set<int_type>::iterator, bool> p;
  p = data_[index].insert(id);
  return p.second;
}

template <typename int_type, std::size_t bucket>
bool static_hashed_set<int_type, bucket>::find(int_type const& id) {
  std::size_t index = calc_index(id);
  boost::lock_guard<boost::mutex> l(static_hashed_set::ref_mutex_[index]);
  return data_[index].find(id) != data_[index].end();
}

template <typename int_type, std::size_t bucket>
bool static_hashed_set<int_type, bucket>::erase(int_type const& id) {
  std::size_t index = calc_index(id);
  boost::lock_guard<boost::mutex> l(static_hashed_set::ref_mutex_[index]);
  if (data_[index].find(id) != data_[index].end()) {
    data_[index].erase(id);
    assert(data_[index].find(id) == data_[index].end());
    return true;
  } else
    return false;
}

template <typename int_type, std::size_t bucket>
std::size_t static_hashed_set<int_type, bucket>::count_nodes(void) const {
  std::size_t r = 0;
  for (int i = 0; i != bucket; ++i) {
    boost::lock_guard<boost::mutex> l(static_hashed_set::ref_mutex_[i]);
    r += data_[i].size();
  }
  return r;
}
