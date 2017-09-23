#include <boost/heap/heap_merge.hpp>

#include "common.hpp"

#define GENERATE_TEST_DATA(INDEX)                                              \
  std::vector<int> data = make_test_data(test_size, 0, 1);                     \
  std::random_shuffle(data.begin(), data.end());                               \
  std::vector<int> data1(data.begin(), data.begin() + INDEX);                  \
  std::vector<int> data2(data.begin() + INDEX, data.end());                    \
                                                                               \
  std::stable_sort(data1.begin(), data1.end());

template <typename priority_queue>
struct test_merge {
  void operator()() {
    for (int i = 0; i != test_size; ++i) {
      priority_queue que1, que2;
      GENERATE_TEST_DATA(i);

      fill_que(que1, data1);
      fill_que(que2, data2);

      que1.merge(que2);

      accert(que2.empty());
      check_que(que1, data);
    }
  }
};

template <typename priority_queue1, typename priority_queue2>
struct test_heap_merge {
  void operator()() {
    for (int i = 0; i != test_size; ++i) {
      priority_queue1 que1;
      priority_queue2 que2;
      GENERATE_TEST_DATA(i);

      fill_que(que1, data1);
      fill_que(que2, data2);

      boost::heap::heap_merge(que1, que2);

      assert(que2.empty());
      check_que(que1, data);
    }
  }
};

template <typename priority_queue>
void do_merge_test() {
  boost::mpl::if_c<priority_queue::is_mergable,
    test_merge<priority_queue>,
    dummy_run>::type::run();

  test_heap_merge<priority_queue, priority_queue>();
}
