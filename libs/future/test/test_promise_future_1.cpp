#include <include/futures.hpp>

auto main() -> decltype(0) {
  int v1 = 0;
  int v2 = 0;
  boost::promise<void> p1;
  boost::promise<void> p2;
  auto f1 = p1.get_future();

  auto waiter = f1.then(
    [&v2, &p2](boost::future<void> future) {
      assert(future.is_ready());
      auto f = boost::async(boost::launch::async,
        [&p2, &v2]() {
          boost::this_thread::sleep_for(boost::chrono::seconds(1));
          v2 = 1;
          p2.set_value();
        });
    });
//  ).then(
//    [&v1, &v2](boost::future<boost::future<void> > future) {
//      assert(future.is_ready());
//      v1 = v2;
//  });

  p1.set_value();
  waiter.wait();
  std::cout << v2 << '\n';
  return 0;
}
