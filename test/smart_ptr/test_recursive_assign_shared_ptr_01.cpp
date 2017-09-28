#include <boost/shared_ptr.hpp>

#include <cassert>

class X {
public:
  static int instances;
  X() { ++instances; }
  ~X() { --instances; }
private:
  X(X const&);
};

int X::instances = 0;

class Y {
public:
  static int instances;
  Y() { ++instances; }
  ~Y() { --instances; }
private:
  Y(Y const&);
};

int Y::instances = 0;
static boost::shared_ptr<void>  global_spv;

class Z {
public:
  static int instances;
  Z() { ++instances; }
  ~Z() {
    --instances;
    boost::shared_ptr<void> pv(new Y);
    global_spv = pv;
  }
private:
  Z(Z const&);
};

int Z::instances = 0;

auto main() -> decltype(0) {
  assert(X::instances == 0);
  assert(Y::instances == 0);
  assert(Z::instances == 0);

  {
    boost::shared_ptr<void> spv(new Z);
    global_spv = spv;
  }

  assert(X::instances == 0);
  assert(Y::instances == 0);
  assert(Z::instances == 1);

  {
    boost::shared_ptr<void> spv(new X);
    global_spv = spv;
  }

  assert(X::instances == 0);
  assert(Y::instances == 1);
  assert(Z::instances == 0);

  global_spv.reset();

  assert(X::instances == 0);
  assert(Y::instances == 0);
  assert(Z::instances == 0);

  return 0;
}
