void* ptr_;
void (*del_)(void*);

template <typename T>
struct secret_class {
public:
  static void test_and_doit(void*);
  static void do_destroy(void* p) { destroy_(p); }
  static void do_throw_ptr_type(void* p) { throw_ptr_type_(p); }
private:
  static void destroy_(void*);
  static void throw_ptr_type_(void*);
};

template <typename T>
T* exact_cast();

#pragma once
#include "../ipp/boundary_crossing.ipp"
