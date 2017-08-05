#include "utils.hpp"

inline boost::xtime delay(int s, int ms = 0, int ns = 0) {
  const int MILLISECONDS_PER_SECOND = 1000;
  const int NANOSECONDS_PER_SECOND = 1000000000;
  const int NANOSECONDS_PER_MILLISECOND = 1000000;

  boost::xtime xt;

  if (boost::TIME_UTC_ != boost::xtime_get(&xt, boost::TIME_UTC_))
    ERROR("boost::timeout_get != boost::TIME_UTC_");

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
