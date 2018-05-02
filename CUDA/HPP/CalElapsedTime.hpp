#ifndef CALELAPSEDTIME_HPP
#define CALELAPSEDTIME_HPP

#include <chrono>
#include <utility>

template <typename TimeType = std::chrono::microseconds>
struct CalElapsedTime {
  template <typename Callable, typename... Args>
  static typename TimeType::rep Execution(Callable&& callable,
    Args&&... args) {
    typedef typename std::chrono::steady_clock clock_type;
    auto start = clock_type::now();
    std::forward<decltype(callable)>(callable)(std::forward<Args>(args)...);
    auto duration =
      std::chrono::duration_cast<TimeType>(clock_type::now() - start);

    return duration.count();
  }
};

#endif
