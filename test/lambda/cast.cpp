#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>

#include <string>

#include <cassert>

class base {
public:
  virtual std::string class_name() const { return "const base"; }
  virtual std::string class_name() { return "base"; }
  virtual ~base() {}
private:
  int x;
};

class derived : public base {
public:
  virtual std::string class_name() const { return "const derived"; }
  virtual std::string class_name() { return "derived"; }
};

void test_1() {
  base* pb = new base;
  derived* pd = new derived;
  base* b = nullptr;
  derived* d = nullptr;

  (boost::lambda::var(b) = boost::lambda::ll_static_cast<base*>(pd))();
  (boost::lambda::var(d) = boost::lambda::ll_static_cast<derived*>(b))();

  assert(b->class_name() == "derived");
  assert(d->class_name() == "derived");

  (boost::lambda::var(b) = boost::lambda::ll_dynamic_cast<derived*>(b))();
  assert(b != nullptr);
  assert(b->class_name() == "derived");

  (boost::lambda::var(d) = boost::lambda::ll_dynamic_cast<derived*>(pb))();
  assert(d == nullptr);

  const derived* pcd = pd;
  assert(pcd->class_name() == "const derived");
  (boost::lambda::var(d) = boost::lambda::ll_const_cast<derived*>(pcd))();
  assert(d->class_name() == "derived");

  int i = 10;
  char* cp = reinterpret_cast<char*>(&i);

  int* ip;
  (boost::lambda::var(ip) = boost::lambda::ll_reinterpret_cast<int*>(cp))();
  assert(*ip == 10);

  assert(std::string(
    boost::lambda::ll_typeid(d)().name()) == std::string(typeid(d).name()));

  assert(boost::lambda::ll_sizeof(boost::lambda::_1)(pd) == sizeof(pd));
  assert(boost::lambda::ll_sizeof(boost::lambda::_1)(*pd) == sizeof(*pd));
  assert(boost::lambda::ll_sizeof(boost::lambda::_1)(pb) == sizeof(pb));
  assert(boost::lambda::ll_sizeof(boost::lambda::_1)(*pb) == sizeof(*pb));

  int arr[100];
  assert(boost::lambda::ll_sizeof(boost::lambda::_1)(arr) == 100*sizeof(int));

  delete pb; delete pd;
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
