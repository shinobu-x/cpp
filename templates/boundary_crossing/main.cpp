#include "hpp/boundary_crossing.hpp"

template <typename T>
struct A {
  void* ptr_;
  void (*throw_)(void*);

  A(T* p) {
    ptr_ = p;
    throw_ = &secret_class<T>::throw_ptr_type_;
  }

  T* cast_throw_xp() const {
    try {
      (*throw_)(ptr_);
    } catch(T* p) {
      return p;
    } catch (...) {
      return 0;
    }
  }
};

auto main() -> decltype(0)
{
  typedef int type_t;
  type_t* a;
  A<type_t>* o1;
  secret_class<A<int>> o2;
  o2.test_and_doit(o1);
  o2.do_destroy(o1); 
 
  return 0;
}
