#include <boost/array.hpp>
#include <boost/thread.hpp>

#include <cassert>

#include "../utils/utils.hpp"

template <bool is_bound = false>
struct queue_stress_test {
  static const unsigned int buckets = 1<<13;
  static const long node_count = 500000;
  const int reader_threads;
  const int writer_threads;
  boost::lockfree::detail::atomic<int> writer_finished;
  static_hashed_set<long, buckets> data;
  static_hashed_set<long, buckets> dequeued;
  boost::array<std::set<long>, buckets> returned;
  boost::lockfree::detail::atomic<int> push_count, pop_count;
  queue_stress_test(int reader, int writer)
    : reader_threads(reader), writer_threads(writer),
      push_count(0), pop_count(0) {}

  template <typename queue>
  void add_items(queue& q) {
    for (long i = 0; i != node_count; ++i) {
      long id = generate_id<long>();
      bool inserted = data.insert(id);
      assert(inserted);

      if (is_bound)
        while (q.bounded_push(id) == false)
          ;
      else
        while (q.push(id) == false)
          ;
      ++push_count;
    }
    writer_finished += 1;
  }

  boost::lockfree::detail::atomic<bool> is_running;

  template <typename queue>
  bool consume_element(queue& q) {
    long id;
    bool r = q.pop(id);

    if (!r)
      return false;

    bool erased = data.erase(id);
    bool inserted = dequeued.insert(id);
    assert(erased);
    assert(inserted);
    ++pop_count;

    return true;
  }

  template <typename queue>
  void get_items(queue& q) {
    for (;;) {
      bool received_element = consume_element(q);
      if (received_element)
        continue;

      if (writer_finished.load() == writer_threads)
        break;
    }
    while (consume_element(q));
  }

  template <typename queue>
  void run(queue& q) {
    writer_finished.store(0);

    boost::thread_group writers;
    boost::thread_group readers;
    assert(q.empty());

    for (int i = 0; i != reader_threads; ++i)
      readers.create_thread(
        boost::bind(&queue_stress_test::template get_items<queue>, this,
        boost::ref(q)));

    for (int i = 0; i != writer_threads; ++i)
      writers.create_thread(
        boost::bind(&queue_stress_test::template add_items<queue>, this,
        boost::ref(q)));

    std::cout << "Thread created\n";

    writers.join_all();

    std::cout << "Writer threads joined, waiting for readers\n";

    readers.join_all();

    std::cout << "Reader threads joined\n";

    assert(data.count_nodes() == (std::size_t)0);
    assert(q.empty());
    assert(push_count == pop_count);
    assert(push_count == (writer_threads * node_count));
  }
};
