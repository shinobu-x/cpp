#include <include/futures.hpp>
#include <exception>
#include <boost/bind.hpp>

void f1(boost::shared_ptr<boost::promise<int> > p) {
  std::cout << __func__ << '\n';
  p->set_value(123);
}

void f2(boost::future<int> f) {
  try {
    std::cout << __func__ << '\n';
    int i = f.get();
    std::cout << i << '\n';
  } catch (std::exception& e) {
    std::cout << e.what() << '\n';
  }
}

auto main() -> decltype(0) {
  try {
    boost::shared_ptr<boost::promise<int> > p(new boost::promise<int>());
    boost::thread t(boost::bind(f1, p));
    boost::future<int> f = p->get_future();
    f.then(boost::launch::deferred, &f2);
    t.join();
  } catch (std::exception& e) {
    std::cout << e.what() << '\n';
  }

  try {
    boost::shared_ptr<boost::promise<int> > p(new boost::promise<int>());
    boost::thread t(boost::bind(f1, p));
    boost::future<int> f = p->get_future();
    f.then(boost::launch::async, &f2);
    t.join();
  } catch (std::exception& e) {
    std::cout << e.what() << '\n';
  }

  return 0;
}
