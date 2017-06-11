#include <iostream>
#include <new>

#define out(x) std::cout << x
#define end    std::cout << '\n'

template <typename T, T N>
struct TEST_T {
  TEST_T(){ std::cout << __func__; }
  T* a = (T*)(malloc(sizeof(T) * N));
};

template <typename T>
T doit() {
  out("p1   "); TEST_T<T, 20>* p1; end;
  out("p2   "); TEST_T<T, 20>* p2 = new TEST_T<T, 20>; end;
  out("p3 1 "); TEST_T<T, 20>* p3 = new (std::nothrow) TEST_T<T, 20>; end;
  out("p3 2 "); new (p3) TEST_T<T, 20>; end;
  out("p4   "); 
  TEST_T<T, 20>* p4 = (TEST_T<T, 20>*) ::operator new (sizeof(TEST_T<T, 20>));
  end;
}
auto main() -> int
{
  doit<int>();
}
