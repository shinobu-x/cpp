#include <algorithm>   /** for_each **/
#include <functional>  /** mem_fn **/
#include <iostream>
#include <thread>
#include <vector>

void worker() {
  std::cout << std::this_thread::get_id() << '\n';
}

template <typename T>
void spawn() {
  std::vector<T> t;
 for (int i=0; i<20; ++i)
   t.push_back(T(worker));

 std::for_each(t.begin(), t.end(), std::mem_fn(&T::join));
}

template <typename T>
void doit() {
  spawn<T>();
}

auto main() -> decltype(0)
{
  doit<std::thread>();
  return 0;
}
