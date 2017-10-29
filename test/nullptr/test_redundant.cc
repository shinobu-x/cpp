#include <cassert>
#include <iostream>
#include <memory>

struct dummy {
  int* a;

  dummy() : a(NULL) {}
};

auto main() -> decltype(0) {

  {
    dummy* d(new dummy);
    assert(d);
    assert(!d->a);
    d = nullptr;
    assert(!d);
  }

  {
    std::shared_ptr<dummy> d(new dummy);
    assert(d);
    assert(!d->a);
    d = nullptr;
    assert(!d);
  }

  return 0;
}
