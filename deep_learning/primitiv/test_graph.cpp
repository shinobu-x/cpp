#include <primitiv/config.h>
#include <primitiv/error.h>
#include <primitiv/functions.h>
#include <primitiv/graph.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/naive_device.h>
#include <primitiv/operator_impl.h>
#include <primitiv/parameter.h>
#include <test_utils.h>

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
    const std::uint32_t fid = node1.operator_id();
    const std::uint32_t vid = node1.value_id();

    primitiv::Node node2 = std::move(node1);
    assert(!node1.valid());
    assert(node2.valid());
    assert(fid == node2.operator_id());
    assert(vid == node2.value_id());

    primitiv::Node node3(std::move(node2));
    assert(!node2.valid());
    assert(node3.valid());
    assert(fid == node3.operator_id());
    assert(vid == node3.value_id());
  }
  {
    primitiv::devices::Naive dev1;
    primitiv::Device::set_default(dev1);
    primitiv::devices::Naive dev2;
    primitiv::Graph g;
    primitiv::Graph::set_default(g);
    typedef typename std::vector<float> value_type;
    typedef primitiv::Node node_type;

    value_type data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    value_type data2 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    value_type data3 {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
    value_type grad(12, 1);

    const node_type node1 = primitiv::functions::input<node_type>(
      primitiv::Shape({2, 2}, 3), data1);
    const node_type node2 = primitiv::functions::input<node_type>(
      primitiv::Shape({2, 2}, 3), data2, dev2);
    const node_type node3 = primitiv::functions::copy(node1, dev2) + node2;

    assert(primitiv::Shape({2, 2}, 3) == node3.shape());
    assert(&dev1 == &node1.device());
    assert(&dev2 == &node2.device());
    assert(&dev2 == &node3.device());
    try {
      g.forward(node3);
    } catch (...) {
      std::cout << __LINE__ << '\n';
    }
    assert(test_utils::vector_match(data1, g.forward(node1).to_vector()));
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
