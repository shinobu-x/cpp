#include <boost/exception_ptr.hpp>
#include <boost/exception/get_error_info.hpp>
#include <boost/thread.hpp>
#include <boost/detail/atomic_count.hpp>

#include <cassert>

typedef boost::error_info<struct tag_answer, int> answer;
boost::detail::atomic_count exc_count(0);

struct
err:
  virtual boost::exception,
  virtual std::exception {
    err() {
      ++exc_count;
    }
    err(err const&) {
      ++exc_count;
    }
    virtual ~err() throw() {
      --exc_count;
    }
  private:
    err& operator=(err const&);
  };

class future {
public:
  future() :
    ready_(false) {}
  void set_exception(boost::exception_ptr const& e) {
    boost::unique_lock<boost::mutex> l(m_);
    exc_ = e;
    ready_ = true;
    cond_.notify_all();
  }
  void get_exception() const {
    boost::unique_lock<boost::mutex> l(m_);
    while (!ready_)
      cond_.wait(l);
    rethrow_exception(exc_);
  }

private:
  bool ready_;
  boost::exception_ptr exc_;
  mutable boost::mutex m_;
  mutable boost::condition_variable cond_;
};

void producer(future& f) {
  f.set_exception(boost::copy_exception(err() << answer(42)));
}

void consumer() {
  future f;
  boost::thread th(boost::bind(&producer, boost::ref(f)));

  try {
    f.get_exception();
  } catch (err& e) {
    int const* ans = boost::get_error_info<answer>(e);
    assert(ans && *ans == 42);
  }

  th.join();
}

void consume() {
  for (int i = 0; i < 100; ++i)
    consumer();
}
void test_1() {
  boost::thread_group threads;

  for (int i = 0; i < 50; ++i)
    threads.create_thread(&consume);

  threads.join_all();
}

void test_2() {
  boost::exception_ptr p = boost::copy_exception(err());

  try {
    rethrow_exception(p);
    assert(false);
  } catch (err&) {
  } catch (...) {
  }
}
  
auto main() -> decltype(0) {
  assert(++exc_count == 1);
  test_1(); test_2();
  assert(!--exc_count);
  return 0;
}
