#include "include/interval_set.h"
#include "include/buffer.h"

#include <list>
#include <map>
#include <set>
#include "rados_model.h"
#include "test_op_stat.h"

void test_op_stat::begin(test_op* in) {
  stat_lock.Lock();
  stats[in->getType()].begin(in);
  stat_lock.Unlock();
}

void test_op_stat::end(test_op* in) {
  stat_lock.Lock();
  stats[in->get_type()].end(in);
  stat_lock.Unlock();
}

void test_op_stat::type_status::export_latencies(
  std::map<double, uint64_t> &in) const {
  std::map<double, uint64_t>::iterator i = in.begin();
  std::multiset<uint64_t>::iterator j = latency.begin();
  int count = 0;

  while (j != latency.end() && i != in.end()) {
    count++;

    if ((((double)count) / ((double)latency.size())) * 100 >= i->first) {
      i->second = *j;
      ++i;
    }
    ++j;
  }
}

std::ostream& operator<<(std::ostream& out, const test_op_stat& rhs) {
  rhs.stat_lock.Lock();

  for (auto i = rhs.stats.begin(); i != rhs.stats.end(); ++i) {
    std::map<double, uint64_t> latency;
    latency[10] = 0;
    latency[50] = 0;
    latency[90] = 0;
    latency[99] = 0;

    i->second.export_latencies(latency);

    out << i->first << " latency: " << std::endl;

    for (std::map<double, uint64_t>::iterator j = latency.begin():
      j != latency.end(); ++j) {
      if (j->second == 0)
        break;

      out << "\t" << j->first << "th percentile: " << j->second / 1000 << "ms"
        << std::endl;

    }
  }

  rhs.stat_lock.Unlock();
  return out;
}
