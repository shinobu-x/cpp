#include <include/futures.hpp>
#include <hpp/basic_future.hpp>

template <typename T>
struct dummy {};

void doit() {
  {
    boost::detail::basic_future<dummy<int> > future;
    boost::exceptional_ptr const e;
    auto ep = future.make_exceptional_future_ptr(e);
  }
  {
    boost::detail::basic_future<dummy<int> > future1;
    boost::detail::basic_future<dummy<int> > future2(future1.future_);
    boost::exceptional_ptr const e;
    boost::detail::basic_future<dummy<int> > future3(e);
    boost::detail::basic_future<dummy<int> > future4 = boost::move(future1);
    boost::detail::basic_future<dummy<int> > future5(boost::move(future1));
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
