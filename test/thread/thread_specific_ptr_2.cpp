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

  const int num_of_thread = 5;
  boost_thread_group threads;

auto main() -> decltype(0) {
  test_1();
  return 0;
}
