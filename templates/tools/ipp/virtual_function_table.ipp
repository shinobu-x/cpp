#include <iostream>

#include "../hpp/virtual_function_table.hpp"

template <typename T>
void local_cast<T>::do_a(void*) {
  std::cout << __func__ << '\n';
}

template <typename T>
void local_cast<T>::do_b(void* b) {
}

template <typename T>
void* local_cast<T>::do_c(void* c) {
}

template <typename T>
void type_t<T>::do_some_a() {
  static const virtual_function_table vft = {
    &local_cast<T>::do_a,
    &local_cast<T>::do_b,
    &local_cast<T>::do_c
  };

  generic_t p;
  p.vft->do_a();
}
