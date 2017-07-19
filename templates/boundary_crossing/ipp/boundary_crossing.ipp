#include <iostream>

#include "../hpp/boundary_crossing.hpp"

template <typename T>
T* exact_cast() {
  return &secret_class<T*>::destroy = del_ ? static_cast<T*>(ptr_) : 0;
}

template <typename T>
void secret_class<T>::test_and_doit(void* p) {
  std::cout << "It works!" << '\n';
}

template <typename T>
void secret_class<T>::destroy_(void* p) {
  delete static_cast<T*>(p);
}

template <typename T>
void secret_class<T>::throw_ptr_type_(void* p) {
  throw static_cast<T*>(p);
}
