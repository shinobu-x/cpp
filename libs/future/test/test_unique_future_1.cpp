#include <include/futures.hpp>

struct async_base {
  virtual ~async_base() {}
  virtual void run() = 0;
};

template <typename T>
struct async_derived : async_base {
private:
  boost::packaged_task<T()> task_;
public:
  async_derived(boost::packaged_task<T()>&& task) :
    task_(boost::move(task)) {}
  ~async_derived() {}

  boost::future<T> get_future() {
    return task_.get_future();
  }

  void run() {
    std::cout << __func__ << '\n';
  }
};

void do_async(async_base* a) {
  std::cout << __func__ << '\n';
  a->run();
}

template <typename F>
boost::future<
  typename boost::result_of<F()>::type> async(F&& f) {
  std::cout << __func__ << '\n';
  typedef typename boost::result_of<F()>::type callback_type;

  async_derived<callback_type>* callback =
    new async_derived<callback_type>(boost::packaged_task<callback_type()>(
      boost::forward<F>(f)));

  boost::future<callback_type> result = callback->get_future();
  do_async(callback);

  return boost::move(result);
}

template <typename F, typename C>
boost::future<
  typename boost::result_of<F(C)>::type> async(F&& f, C&& c) {
  return async(boost::bind(f, c));
}

int f1() {
  return 1;
}

std::size_t f2(const std::string& s) {
  return s.size();
}

void doit() {
  boost::future<int> f = async(&f1);
}

auto main() -> decltype(0) {
  return 0;
}
