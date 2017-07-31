#include <boost/thread.hpp>
#include <boost/thread/tss.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>

struct A {
  void doit() {
    std::cout << "A\n";
    if (!ptr_.get())
      ptr_.reset(new doing());

    for (std::size_t i=0; i<10; ++i)
      ptr_->a += 10; 
  }
private:
  struct doing {
    int a;
    doing() : a(0) {}
  };
  boost::thread_specific_ptr<doing> ptr_;
};

struct B {
  void doit() {
    std::cout << "B\n";
    if (!ptr_.get())
      ptr_.reset(new A());
      ptr_->doit();
  }
private:
  boost::thread_specific_ptr<A> ptr_;
};

struct C {
  void doit() {
    std::cout << "C\n";
    if (!ptr_.get())
      ptr_.reset(new B());
    ptr_->doit();
  }
private:
  boost::thread_specific_ptr<B> ptr_;
};


auto main() -> decltype(0) {
  boost::shared_ptr<C> pc(new C);
  boost::thread ct(&C::doit, pc);
  ct.join();
  return 0;
}
