#include "safe_deleter.hpp"

template <typename T>
void safe_deleter(T*& p) {
  std::cout << __func__ << '\n';
  // compile error if pointer is incomplete type
  typedef char is_complete[sizeof(T) ? 1 : -1];
  (void)sizeof(is_complete);

  delete p;
  p = nullptr;
}
