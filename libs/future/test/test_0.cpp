#include "../hpp/future.hpp"

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
}

auto main() -> decltype(0) {
  return 0;
}
