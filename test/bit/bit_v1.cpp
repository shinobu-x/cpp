#ifdef _WIN32
  #include <intrin.h>
#else
  #include <x86intrin.h>
#endif

#include <iostream>

/* -march=native -mtune=native */

auto main() -> decltype(0)
{
  // A number of 1
  std::cout << _popcnt64(0x00) << '\n';
  std::cout << _popcnt64(0x01) << '\n';
  std::cout << _popcnt64(0x55) << '\n';
  std::cout << _popcnt64(0xffff0000) << '\n';
  std::cout << _popcnt64(0xffffffff) << '\n';

  std::cout << "******" << '\n';

  // A number of 0 from LSb / MSb
  std::cout << _tzcnt_u64(0x00) << '\n';
  std::cout << _tzcnt_u64(0x01) << '\n';
  std::cout << _tzcnt_u64(0x55) << '\n';
  std::cout << _tzcnt_u64(0xffff0000) << '\n';
  std::cout << _tzcnt_u64(0xffffffff) << '\n';

  std::cout << "******" << '\n';

  std::cout << _lzcnt_u64(0x00) << '\n';
  std::cout << _lzcnt_u64(0x01) << '\n';
  std::cout << _lzcnt_u64(0x55) << '\n';
  std::cout << _lzcnt_u64(0xffff0000) << '\n';
  std::cout << _lzcnt_u64(0xffffffff) << '\n';

}
