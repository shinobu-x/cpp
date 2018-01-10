#include "hpp/test_stl_mutex.hpp"

struct test_t1 {
  test_t1() : a(0), b(0), c(0) {}
  int a;
  int b;
  int c;
};

struct test_t2 {
  test_t1 t1;
  read_write_mutex_type m;

  auto do_read() {
    read_lock_type read_lock(m);
    return t1.a + t1.b + t1.c;
  }

  auto do_write(const int x, const int y, const int z) {
    write_lock_type write_lock(m);
    t1.a = x;
    t1.b = y;
    t1.c = z;
  }
};

auto main() -> decltype(0) {
  return 0;
}
