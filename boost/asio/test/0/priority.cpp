#include <boost/asio.hpp>
#include <boost/function.hpp>

#include <iostream>
#include <queue>

/**
 * We user < std::priority_queue > to prioritize queue.
 */
class priority_queue_handler {
public:
  void add(int priority, boost::function<void()> function) {
    handlers_.push(queued_handler(priority, function));
  }

  void execute_all() {
    while (!handlers_.empty()) {
      queued_handler handler = handlers_.top();
      handler.execute();
      handlers_.pop();
    }
  }

  template <typename Handler>
  class wrapped_handler {
  public:
    wrapped_handler(priority_queue_handler& q, int p, Handler h)
      : queue_(q), priority_(p), handler_(h) {}

    void operator()() {
      handler_();
    }

    template <typename A1>
    void operator()(A1 a1) {
      handler_(a1);
    }

    template <typename A1, typename A2>
    void operator()(A1 a1, A2 a2) {
      handler_(a1, a2);
    }

    priority_queue_handler& queue_;
    int priority_;
    Handler handler_;
  };

// ******

  template <typename Handler>
  wrapped_handler<Handler> wrap(int priority, Handler handler) {
    return wrapped_handler<Handler>(*this, priority, handler);
  }

private:
  class queued_handler {
  public:
    queued_handler(int p, boost::function<void()> f)
      : priority_(p), function_(f) {}

    void execute() {
      function_();
    }

    friend bool operator< (const queued_handler& a, const queued_handler& b) {
      return a.priority_ < b.priority_;
    }

  private:
    int priority_;
    boost::function<void()> function_;
  };

  std::priority_queue<queued_handler> handlers_;
};

/**
 * Completion handlers for asynchronous operations are invoked by the io_servic-
 * e associated with the corresponding object.
 *  e.g.,
 *    a socket or deadline_timer
 *
 * Certain guarantees are made on when the handler may be invoked, in particular
 * that a handler can only be invoked from a thread that is currently calling r-
 * un() on the corresponding io_service object.
 * Handlers may subsequently be invoked through other objects that provide addi-
 * tional guarantees.
 *
 * template <typename Function>
 * void asio_handler_invoke(Function function, my_handler* context) {
 *   context->strand_.dispatch(function);
 * }
 */
template <typename Function, typename Handler>
void asio_handler_invoke(Function f,
  priority_queue_handler::wrapped_handler<Handler>* h) {
  h->queue_.add(h->priority_, f);
}

void high_priority(const boost::system::error_code&) {
  std::cout << "High priority job" << '\n';
}
void middle_priority(const boost::system::error_code&) {
  std::cout << "Middle priority job" << '\n';
}
void low_priority() {
  std::cout << "Low priority job" << '\n';
}

auto main() -> decltype(0)
{
  boost::asio::io_service ios;
  priority_queue_handler que;

  // ******
  ios.post(que.wrap(0, low_priority));
  // ******

  boost::asio::ip::tcp::endpoint ep(boost::asio::ip::address_v4::loopback(), 0);
  boost::asio::ip::tcp::acceptor ap(ios, ep);
  boost::asio::ip::tcp::socket ss(ios);

  // ******
  ap.async_accept(ss, que.wrap(100, high_priority));
  // ******

  boost::asio::ip::tcp::socket cs(ios);
  cs.connect(ap.local_endpoint());
  boost::asio::deadline_timer t(ios);
  t.expires_at(boost::posix_time::neg_infin);
  t.async_wait(que.wrap(50, middle_priority));

  // We call < asio_handler_invoke > here
  while (ios.run_one()) {
    while (ios.poll_one())
      ;
    que.execute_all();
  }

  return 0;
}
