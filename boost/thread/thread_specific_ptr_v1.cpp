#include <iostream>

#include <boost/thread.hpp>

template <typename T>
struct type_t {
private:
  std::vector<T> v_;
public:

  void do_something() {
    for (int i=0; i<10; ++i)
      v_.push_back(i);

    show_something();
  }

  // Redundant!
  void show_something() {
    for (typename std::vector<T>::iterator i=v_.begin(); i<v_.end(); ++i)
      std::cout << *i << '\n';
  }
};

template <typename T>
T doit() {
  static boost::thread_specific_ptr<type_t<T> > t;

  if (!t.get())
    t.reset(new type_t<T>);

  t->do_something();
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}

  
