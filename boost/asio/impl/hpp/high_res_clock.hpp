#include <boost/config.hpp>
#include <boost/cstdint.hpp>

#if defined(BOOST_ASIO_WINDOWS)

inline boost::uint64_t high_res_clock() {
  LARGE_INTEGER i;
  QueryPerformanceCounter(&i);
  return i.QuadPart;
}

#elif defined(__GNUC__) && defined(__x86_64__)

inline boost::uint64_t high_res_clock() {
  unsigned long low, high;
  __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high));
  return (((boost::uint64_t)high) << 32) | low;
}

#else

#include <boost/date_time/posix_time/posix_time_type.hpp>

inline boost::uint64_t high_res_clock() {
  boost::posix_time::ptime now =
    boost::posix_time::microsec_clock::universal_time();

  boost::posix_time::ptime epoch(
    boost::gregorian::date(1970, 1, 1),
    boost::posix_time::seconds(0));

  return (now - epoch).total_microseconds();
}

#endif
