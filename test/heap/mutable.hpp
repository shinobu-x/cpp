#ifdef USE_BOOST_RANDOM
#  include <boost/random.hpp>
#else
#  include <cstdlib>
#endif

#include "common.hpp"

#define PUSH_WITH_HANDLES(HANDLES, Q, DATA)                                    \
  std::vector<typename priority_queue::handle_type> HANDLES;                   \
  for (unsigned int k = 0; k != data.size(); ++k)                              \
    HANDLES.push_back(Q.push(DATA[k]));

template <typename priority_queue>
void test_update_decrease() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(i);
    PUSH_WITH_HANDLES(handles, que, data);
    *handles[i] = -1;
    data[i] = -1;
    que.update(handles[i]);
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_update_decrease_function() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    data[i] = -1;
    que.update(handles[i], -1);
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_update_function_identity() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    que.update(handles[i], data[i]);
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_update_shuffled() {
  priority_queue que;
  std::vector<int> data = make_test_data(test_size);
  PUSH_WITH_HANDLES(handles, que, data);
  std::vector<int> shuffled(data);
  std::random_shuffle(shuffled.begin(), shuffled.end());
  for (int i = 0; i != test_size; ++i)
    que.update(handles[i], shuffled[i]);
  check_que(que, data);
}

template <typename priority_queue>
void test_update_increase() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    data[i] = data.back() + 1;
    *handles[i] = data[i];
    que.update(handles[i]);
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_update_increase_function() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    data[i] = data.back() + 1;
    que.update(handles[i], data[i]);
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_decrease() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    *handles[i] = -1;
    data[i] = -1;
    que.decrease(handles[i]);
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_decrease_function() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    data[i] = -1;
    que.decrease(handles[i], -1);
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_decrease_function_identity() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    que.decrease(handles[i], data[i]);
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_increase() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    data[i] = data.back() + 1;
    *handles[i] = data[i];
    que.increase(handles[i]);
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_increase_function() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    que.increase(handles[i], data[i]);
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_increase_function_identity() {
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    que.increase(handles[i], data[i]);
    check_que(que, data);
  }
}

template <typename priority_queue>
void test_erase() {
#ifdef USE_BOOST_RANDOM
  boost::mt19937 rng;
#endif
  for (int i = 0; i != test_size; ++i) {
    priority_queue que;
    std::vector<int> data = make_test_data(test_size);
    PUSH_WITH_HANDLES(handles, que, data);
    for (int j = 0; j != i; ++j) {
#ifdef USE_BOOST_RANDOM
      boost::uniform_int<> range(0, data.size() - 1);
      boost::variable_generator<
        boost::mt19937&, boost::uniform_int<> > gen(rng, range);
      int index = gen();
#else
      int index = std::rand() % (data.size() - 1);
#endif
      que.erase(handles[index]);
      handles.erase(handles.begin() + index);
      data.erase(data.begin() + index);
    }
    std::sort(data.begin(), data.end());
    check_que(que, data);
  }
}

template <typename priority_queue>
void do_test_mutable_heap_update() {
  test_update_increase<priority_queue>();
  test_update_decrease<priority_queue>();
  test_update_shuffled<priority_queue>();
}

template <typename priority_queue>
void do_test_mutable_heap_update_function() {
  test_update_increase_function<priority_queue>();
  test_update_decrease_function<priority_queue>();
  test_update_function_identity<priority_queue>();
}

template <typename priority_queue>
void do_test_mutable_heap_increase() {
  test_increase<priority_queue>();
  test_increase_function<priority_queue>();
  test_increase_function_identity<priority_queue>();
}

template <typename priority_queue>
void do_test_mutable_heap_decrease() {
  test_decrease<priority_queue>();
  test_decrease_function<priority_queue>();
  test_decrease_function_identity<priority_queue>();
}

template <typename priority_queue>
void do_test_mutable_heap_erase() {
  test_erase<priority_queue>();
}

template <typename priority_queue>
void do_test_mutable_heap_handle_from_iterator() {
  priority_queue que;
  que.push(3);
  que.push(1);
  que.push(4);
  que.update(priority_queue::s_handle_from_iterator(que.begin()), 6);
}

template <typename priority_queue>
void do_test_mutable_heap() {
  do_test_mutable_heap_update<priority_queue>();
  do_test_mutable_heap_update_function<priority_queue>();
  do_test_mutable_heap_increase<priority_queue>();
  do_test_mutable_heap_decrease<priority_queue>();
  do_test_mutable_heap_erase<priority_queue>();
  do_test_mutable_heap_handle_from_iterator<priority_queue>();
}

template <typename priority_queue>
void do_test_ordered_iterator() {
  test_ordered_iterators<priority_queue>();
}
