#include <chrono>
#include <thread>
#include <iostream>

#include <ctime>
#include <cstddef>

#include <sys/time.h>

template <typename traits_t>
class basic_timer {
private:
  typedef typename traits_t::time_type tm_t;
  typedef typename traits_t::difference_type diff_t;

  tm_t start_;
  tm_t stop_;

  inline static tm_t now() {
    return traits_t::get_time();
  }

  double elapsed(const tm_t end) const {
    static const tm_t frequency = traits_t::get_freq();
    return double(diff_t(end-start_))/frequency;
  }

public:
  typedef tm_t time_type;
  typedef diff_t difference_type;

  basic_timer() : start_() {}

  difference_type lap() const {
    return now()-start_;
  }

  time_type start() {
    return start_ = now();
  }

  difference_type stop() {
    return (stop_=now())-start_;
  }

  difference_type interval() const {
    return stop_-start_;
  }

  double as_seconds() const {
    return elapsed(stop_);
  }

  double elapsed() const {
    return elapsed(now());
  }
};

// ******

struct clock_time_traits {
  typedef size_t time_type;
  typedef ptrdiff_t difference_type;

  static time_type get_time() {
    time_t t;
    return std::time(&t);
  }

  static time_type get_freq() {
    return 1;
  }
};

// *****

struct cpu_time_traits {
  typedef size_t time_type;
  typedef ptrdiff_t difference_type;

  static time_type get_time() {
    return std::clock();
  }

  static time_type get_freq() {
    return CLOCKS_PER_SEC;
  }
};

// ******

struct bsd_clock_time_traits {
  typedef uint64_t time_type;
  typedef int64_t difference_type;

  static time_type get_time() {
    timeval now;
    gettimeofday(&now, 0);
    return time_type(now.tv_sec)*get_freq()+now.tv_usec;
  }

  static time_type get_freq() {
    return 1000000;
  }
};

template <typename T>
T doit() {
  basic_timer<clock_time_traits> t1;
  basic_timer<cpu_time_traits> t2;
  basic_timer<bsd_clock_time_traits> t3;

  t1.start();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  t1.stop();

  std::cout << t1.as_seconds() << '\n';

  t2.start();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  t2.stop();

  std::cout << t2.as_seconds() << '\n';

  t3.start();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  t3.stop();

  std::cout << t3.as_seconds() << '\n';
}

auto main() -> decltype(0) {
  doit<int>();
  return 0;
}
