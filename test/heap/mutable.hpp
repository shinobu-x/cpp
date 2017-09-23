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
