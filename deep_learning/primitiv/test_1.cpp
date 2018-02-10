#include <primitiv/primitiv.h>

#include <iostream>

void doit() {
  primitiv::devices::Naive dev;
  primitiv::Device::set_default(dev);
  primitiv::Graph g;
  primitiv::Graph::set_default(g);

  /**
   * primitiv::functions::input:
   *   A function which loads values into primitiv. This function require the
   *   following parameters:
   *     - Shape
   *       > Decide data type and size of mini batch.
   *        e.x.,
   *        Shape({3}, 1): 3 dementional vector and 1 mini batch
   *     - value
   *       > List of numbers to be loaded
   *
   *   Return values by this function is object defined as primitiv::Node which
   *   provides virtual results of computation.
   *
   *   Between Nodes and Node and float supports normal mathmatic, and mathmat-
   *   ical functions and functions which control type of Nodes are defined in
   *   primitiv::functions namespace.
   *
   *   Values stored in Node can be retrived by to_vector();
   */

  // x := (1 2 3)
  auto x = primitiv::functions::input<primitiv::Node>(
    primitiv::Shape({3}, 1), {1, 2, 3});
  auto y = 2 * x + 3;
  /**
   * y = 2 * 1 + 3
   * y = 2 * 2 + 3
   * y = 2 * 3 + 3
   */

  for (float v : y.to_vector()) {
    std::cout << v << '\n';
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
