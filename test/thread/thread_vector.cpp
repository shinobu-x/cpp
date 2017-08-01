#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <vector>

#include <cassert>

class thread_group {
public:
  thread_group() {}
  ~thread_group() {
    for (std::list<boost::thread*>::iterator it = threads.begin(),
      end = threads.end();
      it != end;
      ++it) {
      delete *it;
    }
  }

  void interrupt_all() {
    boost::shared_lock<boost::shared_mutex> g(m);

    for (std::list<boost::thread*>::iterator it = threads.begin(),
      end = threads.end();
      it != end;
      ++it)
      (*it)->interrupt();
  }
  
private:
  thread_group(thread_group const&);
  thread_group& operator= (thread_group const&);
  std::list<boost::thread*> threads;
  mutable boost::shared_mutex m;
};

int count = 0;
boost::mutex m;

template <typename T>
void join_all(T& t) {

  for (typename T::iterator it = t.begin(), end = t.end(); it != end; ++it)
    it->join();

}

void do_increment() {
  boost::unique_lock<boost::mutex> l(m);
  ++count;
}

void do_reset() {
  boost::unique_lock<boost::mutex> l(m);
  count = 0;
}

auto main() -> decltype(0) {
  typedef std::vector<boost::thread> vt;

  {
    vt ts;
    ts.reserve(10);
    for (unsigned i = 0; i < 10; ++i) {
      boost::thread t(&do_increment);
      ts.push_back(boost::move(t));
    }
  }

  assert(count == 10);
  do_reset();
  assert(count == 0);

  {
    vt ts;
    ts.reserve(10);
    for (unsigned i = 0; i < 10; ++i)
      ts.emplace_back(&do_increment);
    join_all(ts);
  }

  assert(count == 10);
  do_reset();
  assert(count == 0);
 
  return 0;
}
