#include <boost/thread/condition_variable.hpp>

#include <cassert>

auto main() -> decltype(0) {
  {
    assert(boost::cv_status::no_timeout != boost::cv_status::timeout);
  }
  {
    boost::cv_status cs = boost::cv_status::no_timeout;
    assert(cs == boost::cv_status::no_timeout);
    assert(boost::cv_status::no_timeout == cs);
    assert(cs != boost::cv_status::timeout);
    assert(boost::cv_status::timeout != cs);
  }
  {
    boost::cv_status cs = boost::cv_status::timeout;
    assert(cs == boost::cv_status::timeout);
    assert(boost::cv_status::timeout == cs);
    assert(cs != boost::cv_status::no_timeout);
    assert(boost::cv_status::no_timeout != cs);
  }
  {
    boost::cv_status cs;
    cs = boost::cv_status::no_timeout;
    assert(cs == boost::cv_status::no_timeout);
    assert(boost::cv_status::no_timeout == cs);
    assert(cs != boost::cv_status::timeout);
    assert(boost::cv_status::timeout != cs);
  }
  {
    boost::cv_status cs;
    cs = boost::cv_status::timeout;
    assert(cs == boost::cv_status::timeout);
    assert(boost::cv_status::timeout == cs);
    assert(cs != boost::cv_status::no_timeout);
    assert(boost::cv_status::no_timeout != cs);
  }
  return 0;
}
