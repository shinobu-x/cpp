#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/tss_ptr.hpp>
#include <boost/asio/detail/push_options.hpp>

#include <iostream>

template <typename Owner>
class call_stack_t {
public:
  class context_t : private boost::asio::detail::noncopyable {
  public:
    // Push the owner on to the stack
    explicit context_t(Owner* owner)
      : owner_(owner), next_(call_stack_t<Owner>::top_) {
      call_stack_t<Owner>::top_ = this;
    }
    // Pop the owner from the stack
    ~context_t() {
      call_stack_t<Owner>::top_ = next_;
    }
  private:
    friend class call_stack_t<Owner>;
    Owner* owner_;
    context_t* next_;
  };

  friend class context_t;

  // Determine if the specified owner is on the stack
  static bool is_there(Owner* owner) {
    context_t* ctx = top_;
    while (ctx) {
      if (ctx->owner_ == owner)
        return true;
      ctx = ctx->next_;
    }
    return false;
  }

private:
  static boost::asio::detail::tss_ptr<context_t> top_;
};

template <typename Owner>
boost::asio::detail::tss_ptr<typename call_stack_t<Owner>::context_t>
call_stack_t<Owner>::top_;

struct A {
  template <typename F>
  void do_some(F f) {
    typename call_stack_t<A>::context_t ctx(this);
    f();
  }
};

A a1;

template <typename T>
void doit() {
  if (call_stack_t<T>::is_there(&a1))
    std::cout << "True" << '\n';
  else
    std::cout << "False" << '\n';
}

auto main() -> decltype(0)
{
  a1.do_some(doit<A>);
//  doit();
  return 0;
}
