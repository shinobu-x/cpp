#include <boost/exception_ptr.hpp>
#include <boost/exception/get_error_info.hpp>
#include <boost/thread.hpp>
#include <boost/detail/atomic_count.hpp>

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
  future p;
  boost::thread th(boost::bind(&producer, boost::ref(f)));

  try

auto main() -> decltype(0) {
  return 0;
}
