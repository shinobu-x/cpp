#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/thread/thread.hpp>

#include <cstdlib>
#include <sstream>

typedef boost::interprocess::allocator<int,
  boost::interprocess::managed_shared_memory::segment_manager> shmem_allocator;
typedef boost::lockfree::queue<int, boost::lockfree::allocator<shmem_allocator>,
  boost::lockfree::capacity<2048> > queue;

auto main() -> decltype(0) {
  bool flag = false;

  if (flag) {
    struct shm_remove {
      shm_remove() {
        boost::interprocess::shared_memory_object::remove(
          "boost_queue_interprocess_test_shared_memory"); }

      ~shm_remove() {
        boost::interprocess::shared_memory_object::remove(
          "boost_queue_interprocess_test_shared_memory"); }
    };

    boost::interprocess::managed_shared_memory segment(
      boost::interprocess::create_only,
      "boost_queue_interprocess_test_shared_memory", 262144);

    shmem_allocator alloc_inst(segment.get_segment_manager());

    queue* q = segment.construct<queue>("queue")(alloc_inst);

    for (int i = 0; i != 1024; ++i)
      q->push(i);

    std::string s("queue");
    s += " child";

    if (0 != std::system(s.c_str()))
      return 1;

    while (!q->empty())
      boost::thread::yield();

    return 0;
  } else {
    boost::interprocess::managed_shared_memory segment(
      boost::interprocess::open_only,
      "boost_queue_interprocess_test_shared_memory");

    queue* q = segment.find<queue>("queue").first;

    int from_queue;
    for (int i = 0; i != 1024; ++i) {
      bool success = q->pop(from_queue);
      assert(success);
      assert(from_queue == i);
    }

    segment.destroy<queue>("queue");
  }

  return 0;
}
