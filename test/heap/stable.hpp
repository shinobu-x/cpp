#include <boost/foreach.hpp>
#include "common.hpp"

struct tester {
  tester(int i = 0, int j = 0) : _value(i), _id(j) {}

  bool operator<(tester const& rhs) const {
    return _value < rhs._value;
  }

  bool operator>(tester const& rhs) const {
    return _value > rhs._value;
  }

  bool operator==(tester const& rhs) const {
    return (_value == rhs._value) && (_id == rhs._id);
  }

  int _value, _id;
};

std::ostream& operator<<(std::ostream& out, tester const& t) {
  out << "[" << t._value << " " << t._id << "]";
  return out;
}

std::vector<tester> make_stable_test_data(int size, int same_count = 3,
  int offset = 0, int strive = 1) {
  std::vector<tester> data;

  for (int i = 0; i != size; ++i)
    for (int j = 0; j != same_count; ++j)
      data.push_back(tester(i * strive + offset, j));

  return data;
}

struct compare_by_id {
  bool operator()(tester const& lhs, tester const& rhs) {
    return lhs._id > rhs._id;
  }
};

template <typename priority_queue>
void test_stable_sequential_push() {
  std::vector<tester> data = make_stable_test_data(test_size);

  priority_queue que;

  fill_que(que, data);
  std::stable_sort(data.begin(), data.end(), compare_by_id());
  std::stable_sort(data.begin(), data.end(), std::less<tester>());
  check_que(que, data);
}

template <typename priority_queue>
void test_stable_sequential_reverse_push() {
  std::vector<tester> data1 = make_stable_test_data(test_size);
  priority_queue que;
  std::vector<tester> data2(data1);
  std::stable_sort(data2.begin(), data2.end(), std::greater<tester>());

  fill_que(que, data2);

  std::stable_sort(data1.begin(), data1.end(), compare_by_id());
  std::stable_sort(data1.begin(), data1.end(), std::less<tester>());

  check_que(que, data1);
}

template <typename priority_queue>
void do_stable_heap_test() {
  test_stable_sequential_push<priority_queue>();
  test_stable_sequential_reverse_push<priority_queue>();
}
