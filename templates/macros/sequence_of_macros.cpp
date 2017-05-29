#include <iostream>

#define M_SRT1(a, b) if ((a) > (b)) std::swap((a), (b));

#define M_SRT2(a, b, c) \
  M_SRT1((a), (b)); M_SRT1((a), (c)); M_SRT1((b), (c))

// Fixed version

#define M_SRT3(a, b) if (!((a) > (b))) {} else std::swap((a), (b))

#define M_SRT4(a, b, c) \
  do { M_SRT3((a), (b)); M_SRT3((a), (c)); M_SRT3((b), (c)); } \
  while (false)

template <typename T>
T doit() {
  int a1 = 3, b1 = 7, c1 = 2;
  std::cout << a1 << "/" << b1 << "/" << c1 << '\n';
  if (a1 > 10) M_SRT2(a1, b1, c1);
  std::cout << a1 << "/" << b1 << "/" << c1 << '\n';

/** Expands...
 *  if (a > 10)
 *    M_SRT1(a, b); => if (3 > 7) std::swap(3, 7);
 *    M_SRT1(a, c); => if (3 > 2) std::swap(3, 2); ///< Swap! a is 2
 *    M_SRT1(b, c); => if (7 > 2) std::swap(7, 2); ///< Swap! b is 2
 */

  int a2 = 2, b2 = 5, c2 = 1;
  std::cout << a2 << "/" << b2 << "/" << c2 << '\n';
  if (a2 > 10) { M_SRT1(a2, b2); }
  else { M_SRT1(c2, b2); }
  std::cout << a2 << "/" << b2 << "/" << c2 << '\n';

/** Expands...
 *  if (a > 10)
 *    if (3 < 7) std::swap(3, 7); ///< Swap! a is 7
 *  else
 *    if (2 < 7) std::swap(2, 7); ///< Swap! b is 7
 */

  int a3 = 4, b3 = 2, c3 = 5;
  std::cout << a3 << "/" << b3 << "/" << c3 << '\n';
  M_SRT4(a3, b3, c3);
  std::cout << a3 << "/" << b3 << "/" << c3 << '\n';

}

auto main() -> int
{
  doit<int>();
  return 0;
}
