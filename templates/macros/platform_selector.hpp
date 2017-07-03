#include <iostream>

#include "platform_detector.hpp"

template <typename platform_t>
struct platform_ops;

template <>
struct platform_ops<linux> {
  void whoami() {
    OUT(Linux);
  }
};

template <>
struct platform_ops<windows> {
  void whoami() {
    OUT(Windows);
  }
};

template <>
struct platform_ops<macosx> {
  void whoami() {
    OUT(MAC OS X);
  }
};
