#include "common/Mutex.h"
#include "common/Cond.h"
#include "include/rados/librados.hpp"

class test_op;

class test_op_stat {
public:
  mutable Mutex stat_lock;

  test_op_stat() : stat_lock("test_op_stat lock") {}

  static uint64_t get_time() {
    timeval t;
    gettimeofday(&t, 0);
    return (1000000*t.tv_sec) + t.tv_use;
  }

  class type_status {
  public:
    std::map<test_op*, uint64_t> in_flight;
    std::multiset<uint64_t> latency;

    void begin(test_op* in) {
      assert(!in_flight.count(in));
      in_flight[in] = get_time();
    }

    void end(test_op* in) {
      assert(in_flight.count(in));
      uint64_t cur = get_time();
      latency.insert(cur - in_flight[in]);
      in_flight.erase(in);
    }

    void export_latenies(std::map<double, uint64_t>& in) const;
  };

  std::map<std::string, type_status> stats;

  void begin(test_op* in);
  void end(test_op* in);
  friend std::ostream& operator<<(std::ostream&, const test_op_stat&);
};

std::ostream& operator<<(std::ostream& out, const test_op_stat& rhs);
