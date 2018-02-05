#include "../hpp/future.hpp"

struct test {};

struct callback {
  boost::future<void> operator()(boost::future<void> f) const {
    assert(f.is_ready());
    f.get();
    return boost::make_ready_future();
  }

  boost::future<void> operator()(boost::future<boost::future<void> > f) const {
    assert(f.is_ready());
    f.get();
    return boost::make_ready_future();
  }

};

template <typename T>
void set_thread(boost::promise<T>* p) {
  p->set_value(1);
}

void doit() {
  {
    boost::detail::shared_state_base base_type;
  }

  {
    boost::exceptional_ptr ex;
    boost::detail::shared_state_base base_type(ex);
  }

  {
    boost::detail::shared_state_base base_type;
    auto ex1 = base_type.get_executor();
    boost::executor_ptr_type ex2;
    base_type.set_executor_policy(ex2);
    boost::executor_ptr_type ex3;
    boost::unique_lock<boost::mutex> lock;
    base_type.set_executor_policy(ex3, lock);
  }
  {
    boost::detail::shared_state_base base_type;
    boost::unique_lock<boost::mutex> lock;
    auto b1 = base_type.valid(lock);
    auto b2 = base_type.valid();
    base_type.invalidate(lock);
    base_type.invalidate();
    base_type.validate(lock);
    base_type.validate();
  }
  {
    boost::detail::shared_state_base base_type;
    base_type.set_async();
    base_type.set_deferred();
    base_type.set_executor();
  }
  {
    boost::detail::shared_state_base base_type;
    boost::condition_variable_any cv;
    auto r = base_type.notify_when_ready(cv);
    boost::detail::shared_state_base::notify_when_ready_handle it;
    base_type.unnotify_when_ready(it);
  }
  {
    boost::detail::shared_state_base base_type;
    boost::unique_lock<boost::mutex> lock;
    base_type.do_continuation(lock);
  }
  {
    boost::detail::shared_state<int> shared_state1;
    boost::detail::shared_state<int&> shared_state2;
    boost::detail::shared_state<void> shared_state3;
  }
  {
    boost::detail::future_async_shared_state_base<int> base_type1;
    boost::detail::future_async_shared_state_base<int&> base_type2;
    boost::detail::future_async_shared_state_base<void> base_type3;
  }
  {
    boost::detail::future_async_shared_state<int, test> shared_state1;
    boost::detail::future_async_shared_state<int&, test> shared_state2;
    boost::detail::future_async_shared_state<void, test> shared_state3;
    boost::detail::future_deferred_shared_state<int, test>
      deferred_state1(test());
    boost::detail::future_deferred_shared_state<int&, test>
      deferred_state2(test());
    boost::detail::future_deferred_shared_state<void, test>
      deferred_state3(test());
  }
  {
    boost::detail::future_waiter waiter;
  }
  {
    boost::shared_ptr<boost::detail::shared_state<int> > fp;
    boost::exceptional_ptr const ep;
    boost::detail::basic_future<int> bf1;
    boost::detail::basic_future<int> bf2(fp);
    boost::detail::basic_future<int> bf3(ep);
  }
  {
    boost::promise<int> p;
    boost::future<int> f = p.get_future();
    p.set_value(1);
    auto v = f.get();
    assert(v == 1);
    assert(f.is_ready());
    assert(f.has_value());
    assert(!f.has_exception());
    assert(f.get_state() == boost::future_state::ready);
  }
  {
    boost::promise<int> p;
    boost::future<int> f = p.get_future(); 
    boost::thread(set_thread<int>, &p);
    auto v = f.get();
    assert(v == 1);
    assert(f.is_ready());
    assert(f.has_value());
    assert(!f.has_exception());
    assert(f.get_state() == boost::future_state::ready);
  }
  {
    boost::promise<void> p;
    auto f(p.get_future());
    auto f1 = f.then(callback());
  }
}

auto main() -> decltype(0) {
  return 0;
}
