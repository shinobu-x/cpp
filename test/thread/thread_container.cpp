#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/smart_ptr.hpp>

#include <iostream>
#include <vector>

#include "../utils/utils.hpp"
int count = 0;
boost::mutex m;

namespace {

template <typename TC>
void join_all(TC& tc) {
  for (typename TC::iterator it = tc.begin(); it != tc.end(); ++it)
    *it->join();
}

void increment_count() {
  boost::unique_lock<boost::mutex> l(m);
  std::cout << "count = " << ++count << '\n';
}

template <class T>
struct default_delete {
  typedef T* ptr_type;
  constexpr default_delete() noexcept {}
  template <class U>
  default_delete(const default_delete<U>&) noexcept {}
  void operator()(T* p) const { delete p; }

   default_delete(const default_delete&) = delete;
   default_delete& operator= (const default_delete&) = delete;
   default_delete(default_delete&&) = delete;
   default_delete& operator= (default_delete&&) = delete;
};
} // namespace

auto main() -> decltype(0) {
  {
    typedef boost::shared_ptr<boost::thread> thread_ptr;
    std::list<thread_ptr> threads;

    for (int i = 0; i < 10; ++i)
      threads.push_back(thread_ptr(new boost::thread(&increment_count)));
    assert(threads.size() == 10);

    for (std::list<thread_ptr>::iterator it = threads.begin();
      it != threads.end(); ++it)
      (*it)->join();
  }
  count = 0;
  {
    typedef boost::shared_ptr<boost::thread> thread_ptr;
    std::list<thread_ptr> threads;

    for (int i = 0; i < 10; ++i)
      threads.push_back(thread_ptr(new boost::thread(&increment_count)));
    assert(threads.size() == 10);

    thread_ptr t(new boost::thread(&increment_count));
    threads.push_back(t);
    assert(threads.size() == 11);
    threads.remove(t);
    assert(threads.size() == 10);
    t->join();
    
    for (std::list<thread_ptr>::iterator it = threads.begin();
      it != threads.end(); ++it)
      (*it)->join(); 
  }
  
  return 0;
}
