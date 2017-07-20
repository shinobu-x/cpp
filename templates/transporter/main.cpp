#include "hpp/transporter.hpp"

#include <iostream>
#include <vector>

template <typename T>
static T* transporter_cast(transporter& t) {
  if (wrapper<T>* p = dynamic_cast<wrapper<T>*>(t.ptr_))
    return &(p->obj_);
  else
    return 0;
}

auto main() -> decltype(0)
{
  typedef transporter type_t;
  typedef std::vector<unsigned> container_t;
  type_t a;
  container_t* b;
  b = transporter_cast<container_t>(a);
  
  return 0;
}

