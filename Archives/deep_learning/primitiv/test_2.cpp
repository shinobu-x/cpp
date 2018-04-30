#include <primitiv/primitiv.h>

#include <iostream>

void doit() {

  primitiv::devices::Naive d;
  primitiv::Device::set_default(d);
  primitiv::Graph g;
  primitiv::Graph::set_default(g);

  /**
   *  x := (1 2)
   */
  auto x = primitiv::functions::input<primitiv::Node>({2}, {1, 2});

  /**
   *  A := | 1 2 |
   *       | 1 2 |
   */
  auto a = primitiv::functions::input<primitiv::Node>({2, 2}, {1, 1, 2, 2});

  /**
   *  y := Ax
   */
  auto y = primitiv::functions::matmul(a, x);

  for (auto v : y.to_vector()) {
    std::cout << v << '\n';
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
