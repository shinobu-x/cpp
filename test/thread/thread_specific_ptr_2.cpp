#include <boost/thread/tss.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

boost::mutex m1;
boost::mutex m2;

int tss_instances = 0;
int tss_total = 0;

struct tss_value_t {
  tss_value_t() {
    boost::unique_lock<boost::mutex> l(m2);
    ++tss_instances;
    ++tss_total;
    value_ = 0;
  }

  ~tss_value_t() {
    boost::unique_lock<boost::mutex> l(m2);
    --tss_instances;
  }

  tss_value_t(const tss_value_t&) = delete;
  tss_value_t& operator= (const tss_value_t&) = delete;

  tss_value_t(tss_value_t&&) = delete;
  tss_value_t&& operator= (tss_value_t&&) = delete;
private:
  int value_;
};

auto main() -> decltype(0) {
  return 0;
}
