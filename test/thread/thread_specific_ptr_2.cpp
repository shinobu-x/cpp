#include <boost/thread/tss.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

boost::mutex m1;
boost::mutex m2;

int tss_instances = 0;
int tss_total = 0;

struct tss_value_t {
  tss_value_t() {
    boost::unique_lock<boost::mutex> l(m2);
    ++tss_instances;
    ++tss_total;
    value_ = 0;
  }

  ~tss_value_t() {
    boost::unique_lock<boost::mutex> l(m2);
    --tss_instances;
  }

  tss_value_t(const tss_value_t&) = delete;
  tss_value_t& operator= (const tss_value_t&) = delete;

  tss_value_t(tss_value_t&&) = delete;
  tss_value_t&& operator= (tss_value_t&&) = delete;

  int value_;
};

boost::thread_specific_ptr<tss_value_t> tss_value;

void test_1() {
  tss_value.reset(new tss_value_t());

  for (int i = 0; i < 1000; ++i) {
    int& n = tss_value->value_;
    if (n != 1) {
      boost::unique_lock<boost::mutex> l(m1);
      assert(n == i);
    }
    ++n;
  }
}

typedef pthread_t native_thread_t;

extern "C" {
  void* test_tss_thread_native(void*) {
    test_1();
    return 0;
  }
}

native_thread_t create_native_thread() {
  native_thread_t thread_handle;
  int const r = pthread_create(&thread_handle, 0, &test_tss_thread_native, 0);
  assert(!r);
  return thread_handle;
}

void join_native_thread(native_thread_t thread) {
  void* result = 0;
  int const return_value = pthread_join(thread, &result);
}

void test_2() {
  tss_instances = 0;
  tss_total = 0;

  const int num_of_threads = 5;
  boost::thread_group threads;

  try {
    for (int i = 0; i < num_of_threads; ++i)
      threads.create_thread(&test_1);
    threads.join_all();
  } catch (...) {
    threads.interrupt_all();
    threads.join_all();
    throw;
  }

  std::cout << "tss_instances = " << tss_instances << '\n'
    << "tss_total = " << tss_total << '\n';

  std::cout.flush();

  assert(tss_instances == 0);
  assert(tss_total == 5);

  tss_instances = 0;
  tss_total = 0;

  native_thread_t t1 = create_native_thread();
  native_thread_t t2 = create_native_thread();
  native_thread_t t3 = create_native_thread();
  native_thread_t t4 = create_native_thread();
  native_thread_t t5 = create_native_thread();

  join_native_thread(t1);
  join_native_thread(t2);
  join_native_thread(t3);
  join_native_thread(t4);
  join_native_thread(t5);

  std::cout << "tss_instances = " << tss_instances << '\n'
    << "tss_total = " << tss_total << '\n';
  std::cout.flush();

  assert(tss_instances == 0);
  assert(tss_total == 5);

}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}
