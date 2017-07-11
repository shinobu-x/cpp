#include <iostream>

template <typename T, bool SMALL_OBJECT = (sizeof(T)<sizeof(void*))>
struct clone_of;

template <typename T>
struct clone_of<T, true> {
  typedef T type;
};

template <typename T>
struct clone_of<T, false> {
  typedef const T& type;
};

template <typename static_type/*, typename aux_t = void */>
class static_interface {
public:
  typedef static_type type;

  typename clone_of<static_type>::type clone() const {
    return true_this();
  }

protected:
  static_interface(){}
  ~static_interface(){}

  static_type& true_this() {
    return static_cast<static_type&>(*this);
  }

  const static_type& true_this() const {
    return static_cast<const static_type&>(*this);
  }
};

template <typename static_type>
class type_t : public static_interface<static_type> {
protected:
  ~type_t(){}

public:
  typedef double value_type;

  value_type do_some() const {
    return 0;
  }
};

template <typename T>
void doit() {
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
