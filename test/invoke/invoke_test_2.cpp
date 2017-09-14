#include <boost/thread/detail/invoke.hpp>

int count = 0;

void f0(int a) {
  count += a;
}

struct test_invoke_2 {
  test_invoke_2() : data(0) {}

  void operator()(int a) {
    count += a;
  }

  void operator()(int) const;
  void operator()(int, int) const;
  void operator()(int, int, int) const;
  void operator()(int, int, int, int) const;
  void operator()(int, int, int, int, int) const;
  void operator()(int, int, int, int, int, int) const;

  void f1() { ++count; }
  void f2() const { ++(++count); }

  int data;
};

void test_invoke_2::operator()(int a) const {
  count += ++a;
}

void test_invoke_2::operator()(int a, int b) const {
  count += (a + b);
}

void test_invoke_2::operator()(int a, int b, int c) const {
  count += (a + b + c);
}

void test_invoke_2::operator()(int a, int b, int c, int d) const {
  count += (a + b + c + d);
}

void test_invoke_2::operator()(int a, int b, int c, int d, int e) const {
  count += (a + b + c + d + e);
}

void test_invoke_2::operator()(int a, int b, int c, int d ,int e, int f) const {
  count += (a + b + c + d + e + f);
}

void do_test_invoke_2() {
  const int i = 1; 
  void (*fp0)(int) = f0;
  int save = count;
  test_invoke_2 obj;
  test_invoke_2* ptr = &obj;
  void (test_invoke_2::*fp1)() = &test_invoke_2::f1;
  void (test_invoke_2::*fp2)() const = &test_invoke_2::f2;
  const test_invoke_2 cobj;

  {
    boost::detail::invoke(f0, i);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(f0, i);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(fp0, i);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(fp0, i);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(obj, i);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(obj, i);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(cobj, i);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke(cobj, i, i);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke(cobj, i, i, i);
    assert(count == save + (i*3));
    save = count;
  }
  {
    boost::detail::invoke(cobj, i, i, i, i);
    assert(count == save + (i*4));
    save = count;
  }
  {
    boost::detail::invoke(cobj, i, i, i, i, i);
    assert(count == save + (i*5));
    save = count;
  }
  {
    boost::detail::invoke(cobj, i, i, i, i, i, i);
    assert(count == save + (i*6));
    save = count;
  }
  {
    boost::detail::invoke<void>(cobj, i);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke<void>(cobj, i, i);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke<void>(cobj, i, i, i);
    assert(count == save + (i*3));
    save = count;
  }
  {
    boost::detail::invoke<void>(cobj, i, i, i, i);
    assert(count == save + (i*4));
    save = count;
  }
  {
    boost::detail::invoke<void>(cobj, i, i, i, i, i);
    assert(count == save + (i*5));
    save = count;
  }
  {
    boost::detail::invoke<void>(cobj, i, i, i, i, i, i);
    assert(count == save + (i*6));
    save = count;
  }
  {
    boost::detail::invoke(fp1, obj);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(fp1, obj);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(fp1, ptr);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(fp1, ptr);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(fp2, obj);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke<void>(fp2, obj);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke(fp2, ptr);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke<void>(fp2, ptr);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke(test_invoke_2(), i);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(test_invoke_2(), i);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(&test_invoke_2::f1, obj);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(&test_invoke_2::f1, obj);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(&test_invoke_2::f1, ptr);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(&test_invoke_2::f1, ptr);
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(&test_invoke_2::f1, test_invoke_2());
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke<void>(&test_invoke_2::f1, test_invoke_2());
    assert(count == save + i);
    save = count;
  }
  {
    boost::detail::invoke(&test_invoke_2::f2, obj);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke<void>(&test_invoke_2::f2, obj);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke(&test_invoke_2::f2, ptr);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke<void>(&test_invoke_2::f2, ptr);
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke(&test_invoke_2::f2, test_invoke_2());
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke<void>(&test_invoke_2::f2, test_invoke_2());
    assert(count == save + (i*2));
    save = count;
  }
  {
    boost::detail::invoke(&test_invoke_2::data, obj) = 1;
    assert(boost::detail::invoke(&test_invoke_2::data, obj) == 1);
  }
  {
    boost::detail::invoke<int>(&test_invoke_2::data, obj) = 2;
    assert(boost::detail::invoke<int>(&test_invoke_2::data, obj) == 2);
  }
  {
    boost::detail::invoke(&test_invoke_2::data, ptr) = 3;
    assert(boost::detail::invoke(&test_invoke_2::data, ptr) == 3);
  }
  {
    boost::detail::invoke<int>(&test_invoke_2::data, ptr) = 4;
    assert(boost::detail::invoke<int>(&test_invoke_2::data, ptr) == 4);
  }
}

auto main() -> decltype(0) {
  do_test_invoke_2();
  return 0;
}
