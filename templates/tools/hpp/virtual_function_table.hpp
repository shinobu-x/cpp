struct virtual_function_table {
  void (*do_a)(void*);
  void (*do_b)(void*);
  void* (*do_c)(void*);
};

template <typename T>
struct local_cast {
  static void do_a(void*);
  static void do_b(void*);
  static void* do_c(void*);
};

struct generic_t {
  void (*f_a)(void*);
  void (*f_b)(void*);
  void* (*f_c)(void*);
  const virtual_function_table* vft;
};

template <typename T>
class type_t : generic_t {
public:
  void do_some_a();
  void do_some_b();
  void do_some_c();
};

#pragma once
#include "../ipp/virtual_function_table.ipp"
