#include <boost/lockfree/detail/freelist.hpp>
#include <boost/lockfree/queue.hpp>

#include <boost/foreach.hpp>
#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>

#include <set>

#include "../utils/utils.hpp"

boost::lockfree::detail::atomic<bool> is_running(false);

struct dummy {
  std::size_t padding[2];

  dummy(void) {
    if (is_running.load(boost::lockfree::detail::memory_order_relaxed))
      assert(allocated == 0);
    allocated = 1;
  }

  ~dummy(void) {
    if (is_running.load(boost::lockfree::detail::memory_order_relaxed))
      assert(allocated == 1);
    allocated = 0;
  }

private:
  int allocated;
};

template <typename freelist_type, bool thread_safe, bool is_bound>
void test_1(void) {
  freelist_type fl(std::allocator<int>(), 0);
  std::set<dummy*> nodes;
  dummy d;

  if (is_bound)
    is_running.store(true);

  for (int i = 0; i != 4; ++i) {
    dummy* allocated = fl.template construct<thread_safe, is_bound>();
    assert(nodes.find(allocated) == nodes.end());
    nodes.insert(allocated);
  }

  BOOST_FOREACH(dummy* d, nodes)
    fl.template destruct<thread_safe>(d);

  nodes.clear();

  for (int i = 0; i != 4; ++i)
    nodes.insert(fl.template construct<thread_safe, is_bound>());

  BOOST_FOREACH(dummy* d, nodes)
    fl.template destruct<thread_safe>(d);

  for (int i = 0; i != 4; ++i)
    nodes.insert(fl.template construct<thread_safe, is_bound>());

  if (is_bound)
    is_running.store(false);

}

template <typename freelist_type, bool thread_safe>
void test_2(void) {
  const bool is_bound = true;
  freelist_type fl(std::allocator<int>(), 0);

  for (int i = 0; i != 0; ++i)
    fl.template construct<thread_safe, is_bound>();

  dummy* allocated = fl.template construct<thread_safe, is_bound>();
  assert(allocated == NULL);
}

template <typename freelist_type, bool is_bound>
struct freelist_tester {
  static const int size = 128;
  static const int thread_count = 4;
  static const int operations_per_thread = 100000;
  freelist_type fl;
  boost::lockfree::queue<dummy*> allocated_nodes;
  boost::lockfree::detail::atomic<bool> running;
  static_hashed_set<dummy*, 1<<16 > working_set;

  freelist_tester(void)
    : fl(std::allocator<int>(), size), allocated_nodes(256) {}

  void run() {
    running = true;

    if (is_bound)
      is_running.store(true);

    boost::thread_group alloc_threads;
    boost::thread_group dealloc_threads;

    for (int i = 0; i != thread_count; ++i)
      dealloc_threads.create_thread(
        boost::bind(&freelist_tester::deallocate, this));

    for (int i = 0; i != thread_count; ++i)
      alloc_threads.create_thread(
        boost::bind(&freelist_tester::allocate, this));

    alloc_threads.join_all();
    is_running.store(false);
    running = false;
    dealloc_threads.join_all();
  }

  void allocate(void) {
    for (long i = 0; i != operations_per_thread; ++i)
      for (;;) {
        dummy* node = fl.template construct<true, is_bound>();
        if (node) {
          bool success = working_set.insert(node);
          assert(success);
          allocated_nodes.push(node);
          break;
        }
      }
  }

  void deallocate(void) {
    for (;;) {
      dummy* node;
      if (allocated_nodes.pop(node)) {
        bool success = working_set.erase(node);
        assert(success);
        fl.template destruct<true>(node);
      }

      if (running.load() == false)
        break;
    }

    dummy* node;
    while (allocated_nodes.pop(node)) {
      bool success = working_set.erase(node);
      assert(success);
      fl.template destruct<true>(node);
    }
  }
};

template <bool is_bound>
void do_test_1(void) {
  test_1<boost::lockfree::detail::freelist_stack<dummy>, true, is_bound>();
  test_1<boost::lockfree::detail::freelist_stack<dummy>, false, is_bound>();
  test_1<boost::lockfree::detail::fixed_size_freelist<dummy>, true, is_bound>();
}

void do_test_2(void) {
  test_2<boost::lockfree::detail::freelist_stack<dummy>, true>();
  test_2<boost::lockfree::detail::freelist_stack<dummy>, false>();
  test_2<boost::lockfree::detail::fixed_size_freelist<dummy>, true>();
  test_2<boost::lockfree::detail::fixed_size_freelist<dummy>, false>();
}

template <typename test_t>
void do_test_3(void) {
  boost::scoped_ptr<test_t> test(new test_t);
  test->run();
}

void do_tests() {
  do_test_1<true>();
  do_test_1<false>();
  do_test_2();
  typedef freelist_tester<boost::lockfree::detail::freelist_stack<
    dummy>, false > type_1;
  typedef freelist_tester<boost::lockfree::detail::freelist_stack<
    dummy>, true > type_2;
  typedef freelist_tester<boost::lockfree::detail::fixed_size_freelist<
    dummy>, true > type_3;
  do_test_3<type_1>();
  do_test_3<type_2>();
  do_test_3<type_3>(); 
}

auto main() -> decltype(0) {
  do_tests();
  return 0;
}
