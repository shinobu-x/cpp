#include "../include/futures.hpp"
#include "../hpp/shared_state_base.hpp"

void doit() {
  {
    boost::exceptional_ptr e;
    boost::detail::shared_state_base state_base(e);
  }
  {
    boost::detail::shared_state_base state_base;
    boost::shared_ptr<boost::executor> ex = state_base.get_executor();
    state_base.set_executor_policy(ex);
  }
  {
    boost::detail::shared_state_base state_base;
    boost::shared_ptr<boost::executor> ex = state_base.get_executor();
    boost::mutex m;
    boost::unique_lock<boost::mutex> lock(m);
    state_base.set_executor_policy(ex, lock);
  }
  {
    boost::detail::shared_state_base state_base;
    boost::shared_ptr<boost::executor> ex = state_base.get_executor();
    boost::mutex m;
    boost::lock_guard<boost::mutex> lock(m);
    state_base.set_executor_policy(ex, lock);
  }
  {
    boost::detail::shared_state_base state_base;
    boost::mutex m;
    boost::unique_lock<boost::mutex> lock(m);
    assert(state_base.valid(lock));
    assert(!state_base.valid());
    state_base.invalidate(lock);
    state_base.invalidate();
    state_base.validate(lock);
    state_base.validate();
    state_base.set_async();
    assert(!state_base.is_deferred_);
    state_base.set_deferred();
    assert(state_base.is_deferred_);
    state_base.set_executor();
    assert(!state_base.is_deferred_);
  }
  {
    boost::detail::shared_state_base state_base;
    boost::condition_variable_any cv;
    std::list<boost::condition_variable_any*>::iterator it =
      state_base.notify_when_ready(cv);
    state_base.unnotify_when_ready(it);
  }
  {
    boost::detail::shared_state_base state_base;
    boost::mutex m;
    boost::unique_lock<boost::mutex> lock(m);
    state_base.do_continuation(lock);
    boost::shared_ptr<boost::detail::shared_state_base> c;
    state_base.set_continuation_ptr(c, lock);
  }
  {
    boost::detail::shared_state_base state_base;
    state_base.notify_deferred();
    boost::mutex m;
    boost::unique_lock<boost::mutex> lock(m);
    state_base.mark_finished_internal(lock);
    state_base.do_callback(lock);
    state_base.is_deferred_ = true;
    state_base.run_if_is_deferred();
    assert(!state_base.is_deferred_);
    state_base.is_deferred_ = true;
    state_base.run_if_is_deferred_or_ready();
    assert(!state_base.is_deferred_);
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
