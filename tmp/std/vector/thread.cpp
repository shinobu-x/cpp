#include <iostream>
#include <thread>
#include <vector>

template <typename T, T N, typename U>
struct type_t {
public:
  T get() {
    return a_;
  }

  void set(T* v) {
    T a;
    a_ = (*v)*(*v);
  }

  void th() {
    U t([]{ std::cout << std::this_thread::get_id() << '\n'; });
    t.join();
  }

private:
  T a_;
};

template <typename T, T N>
T doit() {
  std::vector<type_t<T, 10, std::thread> > vt;
  type_t<T, 10, std::thread> t;
  T a = N;
  vt.push_back(t);
  for (typename std::vector<type_t<T, 10, std::thread> >::iterator it=
    vt.begin(); it!=vt.end(); ++it) {
    it->set(&a);
    std::cout << it->get() << '\n';
    it->th();
  }
}

auto main() -> int
{
  doit<int, 10>();
  return 0;
}
