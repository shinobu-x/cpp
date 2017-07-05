#include <iostream>
#include <type_traits>

void* condition;

template <void*>
struct selector {};

template <typename T,
  typename std::enable_if<std::is_class<T>::value>::type*& = condition >
bool is_class() { return true; }

template <typename T, 
  typename std::enable_if<!std::is_class<T>::value>::type*& = condition >
bool is_class() { return false; }

template <typename T>
void doit() {
  if (is_class<T>())
    std::cout << "Class" << '\n';
  else
    std::cout << "Non Class" << '\n';
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
