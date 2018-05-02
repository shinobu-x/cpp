#ifndef INITDATA_HPP
#define INITDATA_HPP

#include <random>

template <typename ValueType = float>
inline void InitData(ValueType* data, int size) {
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_int_distribution<> urnd(0, 999);

  for (int i = 0; i < size; ++i) {
    data[i] = (ValueType)(urnd(mt) & 0xff) / 10.0f;
  }

}

#endif
