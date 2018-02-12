#include <primitiv/config.h>
#include <primitiv/error.h>
#include <primitiv/functions.h>
#include <primitiv/graph.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/naive_device.h>
#include <primitiv/operator_impl.h>
#include <primitiv/parameter.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>

void doit() {
  primitiv::devices::Naive dev1;
  primitiv::devices::Naive dev2;

  try {
    primitiv::Graph::get_default();
  } catch (...) {
    std::cout << "Device is not set\n";
  }
  {
    primitiv::Graph g1;
    primitiv::Graph::set_default(g1);
    assert(&g1 == &primitiv::Graph::get_default());

    primitiv::Graph g2;
    primitiv::Graph::set_default(g2);
    assert(&g1 != &primitiv::Graph::get_default());
    assert(&g2 == &primitiv::Graph::get_default());

    primitiv::Graph g3;
    primitiv::Graph::set_default(g3);
    assert(&g1 != &primitiv::Graph::get_default());
    assert(&g2 != &primitiv::Graph::get_default());
    assert(&g3 == &primitiv::Graph::get_default());
  }
  try {
    primitiv::Graph::get_default();
  } catch (...) {
    std::cout << "No default\n";
  }
  {
    primitiv::Node node;
    assert(!node.valid());
    try { node.graph();
    } catch (...) {
      std::cout << "Graph is not set\n";
    }
    try {
      node.operator_id();
    } catch (...) {
      std::cout << "Operator id is not set\n";
    }
    try {
      node.value_id();
    } catch (...) {
      std::cout << "Value id is not set\n";
    }
    try {
      node.shape();
    } catch (...) {
      std::cout << "Shape is not set\n";
    }
    try {
      node.device();
    } catch (...) {
      std::cout << "Device is not set\n";
    }
    try {
      node.to_float();
    } catch (...) {
      std::cout << "Value is not set\n";
    }
    try {
      node.to_vector();
    } catch (...) {
      std::cout << "Value is not set\n";
    }
    try {
      node.backward();
    } catch (...) {
      std::cout << "Backward is not set\n";
    }
  }
  {
    primitiv::devices::Naive dev;
    primitiv::Device::set_default(dev);
    assert(&dev == &primitiv::Device::get_default());

    primitiv::Graph g;
    primitiv::Graph::set_default(g);
    assert(&g == &primitiv::Graph::get_default());

    primitiv::Node node1 = primitiv::functions::zeros<primitiv::Node>({2, 2});
    assert(node1.valid());
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
