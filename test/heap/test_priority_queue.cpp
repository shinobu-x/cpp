#include <boost/heap/priority_queue.hpp>

#include <algorithm>

#include "common.hpp"
#include "stable.hpp"
#include "merge.hpp"

template <bool is_stable>
void do_common_test() {
  typedef boost::heap::priority_queue<
    int, boost::heap::stable<is_stable> > priority_queue;

  do_common_heap_test<priority_queue>();
  do_iterator_heap_test<priority_queue>();
  do_copyable_heap_test<priority_queue>();
  do_moveable_heap_test<priority_queue>();
  do_merge_test<priority_queue>(); 

  if (is_stable) {
    typedef boost::heap::priority_queue<
      tester, boost::heap::stable<is_stable> > stable_queue;
    do_stable_heap_test<stable_queue>();
  }
}

auto main() -> decltype(0) {
  do_common_test<false>();
  do_common_test<true>();
  return 0;
}
