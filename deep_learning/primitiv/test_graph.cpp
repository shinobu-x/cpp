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

    assert(data1.size() == g.forward(node1).to_vector().size());
    assert(data1.size() == node1.to_vector().size());
    assert(data2.size() == g.forward(node2).to_vector().size());
    assert(data2.size() == node2.to_vector().size());
    assert(data3.size() == g.forward(node3).to_vector().size());
    assert(data3.size() == node3.to_vector().size());

    for (int i = 0; i < data1.size(); ++i) {
      assert(data1[i] == g.forward(node1).to_vector()[i]);
    }
    for (int i = 0; i < data2.size(); ++i) {
      assert(data2[i] == g.forward(node2).to_vector()[i]);
    }
    for (int i = 0; i < data3.size(); ++i) {
      assert(data3[i] == g.forward(node3).to_vector()[i]);
    }
  }
  {
    primitiv::devices::Naive dev1;
    primitiv::Device::set_default(dev1);
    primitiv::devices::Naive dev2;
    primitiv::Graph g;
    primitiv::Graph::set_default(g);

    const std::vector<float> dummy(12);
    const primitiv::Node node1 = primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2, 2}, 3), dummy);
    const primitiv::Node node2 = primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2, 2}, 3), dummy, dev2);
    const primitiv::Node node3 = node1 + node2;

    try {
      g.forward(node3);
    } catch (std::exception& e) {
      std::cout << e.what() << '\n';
    }
  }
  {
    primitiv::devices::Naive dev1;
    primitiv::Device::set_default(dev1);
    primitiv::Graph g;
    primitiv::Graph::set_default(g);
    assert(0u == g.num_operators());

    {
      primitiv::functions::input<primitiv::Node>({}, {1});
      primitiv::functions::input<primitiv::Node>({}, {1});
      assert(2u == g.num_operators());
    }

    g.clear();
    assert(0u == g.num_operators());

    {
      primitiv::functions::input<primitiv::Node>({}, {1});
      primitiv::functions::input<primitiv::Node>({}, {1});
      primitiv::functions::input<primitiv::Node>({}, {1});
      assert(3u == g.num_operators());
    }
  }
  {
    primitiv::devices::Naive dev;
    primitiv::Device::set_default(dev);
    primitiv::Graph g;
    primitiv::Graph::set_default(g);

    const std::vector<float> data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    const std::vector<float> data2 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    std::vector<primitiv::Node> nodes;

    nodes.emplace_back(primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2, 2}, 3), data1));
    nodes.emplace_back(primitiv::functions::ones<primitiv::Node>({2, 2}));
    nodes.emplace_back(primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2, 2}, 3), data2));
    nodes.emplace_back(nodes[0] + nodes[1]);
    nodes.emplace_back(nodes[1] - nodes[2]);
    nodes.emplace_back(nodes[3] * nodes[4]);
    nodes.emplace_back(nodes[5] + 1);
    nodes.emplace_back(primitiv::functions::sum(nodes[6], 0));
    nodes.emplace_back(primitiv::functions::sum(nodes[7], 1));
    nodes.emplace_back(primitiv::functions::batch::sum(nodes[8]));

    assert(10u == nodes.size());
    assert(10u == g.num_operators());

    std::cout << g.dump("dot");

    const std::vector<primitiv::Shape> expected_shapes {
      primitiv::Shape({2, 2}, 3),
      {2, 2},
      primitiv::Shape({2, 2}, 3),
      primitiv::Shape({2, 2}, 3),
      primitiv::Shape({2, 2}, 3),
      primitiv::Shape({2, 2}, 3),
      primitiv::Shape({2, 2}, 3),
      primitiv::Shape({1, 2}, 3),
      primitiv::Shape({}, 3),
      {},
    };

    for (int i = 0; i < nodes.size(); ++i) {
      assert(expected_shapes[i] == nodes[i].shape());
      assert(&dev == &nodes[i].device());
    }

    g.forward(nodes.back());

    const std::vector<std::vector<float> > expected_values {
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
      {1, 1, 1, 1},
      {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
      {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5},
      {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1},
      {2, 3, 4, 5, 0, 0, 0, 0, -2, -3, -4, -5},
      {3, 4, 5, 6, 1, 1, 1, 1, -1, -2, -3, -4},
      {7, 11, 2, 2, -3, -7},
      {18, 4, -10},
      {12},
    };

    for (int i = 0; i < nodes.size(); ++i) {
      const primitiv::Tensor& v = g.forward(nodes[i]);
      assert(v.valid());
    }
    for (int i = 0; i < nodes.size(); ++i) {
      const primitiv::Tensor& v = g.forward(nodes[i]);
      for (int j = 0; j < expected_values[i].size(); ++j) {
        assert(expected_values[i][j] == v.to_vector()[j]);
        assert(expected_values[i][j] == nodes[i].to_vector()[j]);
      }
    }
  }
  {
    primitiv::devices::Naive dev;
    primitiv::Device::set_default(dev);

    primitiv::Parameter w1({2, 2}, {1, -1, 1, -1});
    primitiv::Parameter b1({2}, {-1, -1});
    primitiv::Parameter w2({1, 2}, {1, 1});
    primitiv::Parameter b2({}, {1});

    const std::vector<float> inputs {1, 1, 1, -1, -1, 1, -1, -1};
    const std::vector<float> outputs {1, -1, -1, 1};

    primitiv::Graph g;
    primitiv::Graph::set_default(g);

    std::vector<primitiv::Node> nodes;

    nodes.emplace_back(primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2}, 4), inputs));
    nodes.emplace_back(primitiv::functions::parameter<primitiv::Node>(w1));
    nodes.emplace_back(primitiv::functions::parameter<primitiv::Node>(b1));
    nodes.emplace_back(primitiv::functions::parameter<primitiv::Node>(w2));
    nodes.emplace_back(primitiv::functions::parameter<primitiv::Node>(b2));

    nodes.emplace_back(primitiv::functions::matmul(nodes[1], nodes[0]));
    nodes.emplace_back(nodes[5] + nodes[2]);
    nodes.emplace_back(primitiv::functions::tanh(nodes[6]));
    nodes.emplace_back(primitiv::functions::matmul(nodes[3], nodes[7]));
    nodes.emplace_back(nodes[8] + nodes[4]);

    nodes.emplace_back(primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({}, 4), outputs));
    nodes.emplace_back(nodes[9] - nodes[10]);
    nodes.emplace_back(nodes[11] * nodes[11]);
    nodes.emplace_back(primitiv::functions::batch::sum(nodes[12]));

    assert(nodes.size() == g.num_operators());

    std::cout << g.dump("dot");
    g.forward(nodes.back());

    const float h1 =   .76159416;
    const float h2 =   .99505475;
    const float h3 =  -.23346060;
    const float h4 = -1.5231883;
    const float h5 =   .76653940;
    const float h6 =  -.52318831;
    const float h7 =   .47681169;

    const std::vector<std::vector<float> > expected_values {
      {1, 1, 1, -1, -1, 1, -1, -1},
      {1, -1, 1, -1},
      {-1, -1},
      {1, 1},
      {1},
      {2, -2, 0, 0, 0, 0, -2, 2},
      {1, -3, -1, -1, -1, -1, -3, 1},
      {h1, -h2, -h1, -h1, -h1, -h1, -h2, h1},
      {h3, h4, h4, h3},
      {h5, h6, h6, h5},
      {1, -1, -1, 1},
      {h3, h7, h7, h3},
      {h3 * h3, h7 * h7, h7 * h7, h3 * h3},
      {2 * (h3 * h3 + h7 * h7)},
    };

    for (int i = 0; i < nodes.size(); ++i) {
      const primitiv::Tensor& v = g.forward(nodes[i]);
      assert(v.valid());
    }

    for (int i = 0; i < nodes.size(); ++i) {
      const primitiv::Tensor& v = g.forward(nodes[i]);
      for (int j = 0; j < expected_values[i].size(); ++j) {
        //assert(expected_values[i][j] == v.to_vector()[j]);
        //assert(expected_values[i][j] == nodes[i].to_vector()[j]);
      }
    }
  }

  {
    primitiv::devices::Naive dev;
    primitiv::Device::set_default(dev);

    primitiv::Parameter wix({2, 2}, {.3, .1, .5, .3});
    primitiv::Parameter wfx({2, 2}, {.4, .1, .5, .8});
    primitiv::Parameter wox({2, 2}, {.5, .9, .9, .7});
    primitiv::Parameter wjx({2, 2}, {.2, .6, .9, .3});
    primitiv::Parameter wih({2, 2}, {.2, .3, .3, .3});
    primitiv::Parameter wfh({2, 2}, {.8, .4, .8, .3});
    primitiv::Parameter woh({2, 2}, {.6, .2, .2, .7});
    primitiv::Parameter wjh({2, 2}, {.6, .4, .9, .5});
    primitiv::Parameter bi({2}, primitiv::initializers::Constant(0));
    primitiv::Parameter bf({2}, primitiv::initializers::Constant(0));
    primitiv::Parameter bo({2}, primitiv::initializers::Constant(0));
    primitiv::Parameter bj({2}, primitiv::initializers::Constant(0));

    primitiv::Graph g;
    primitiv::Graph::set_default(g);

    const primitiv::Node x = primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2}, 2), {2, -2, 0.5, -0.5});
    const primitiv::Node h = primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2}, 2), {-1, 1, -0.5, 0.5});
    const primitiv::Node c = primitiv::functions::zeros<primitiv::Node>({2});

    const primitiv::Node nwix =
      primitiv::functions::parameter<primitiv::Node>(wix);
    const primitiv::Node nwfx =
      primitiv::functions::parameter<primitiv::Node>(wfx);
    const primitiv::Node nwox =
      primitiv::functions::parameter<primitiv::Node>(wox);
    const primitiv::Node nwjx =
      primitiv::functions::parameter<primitiv::Node>(wjx);
    const primitiv::Node nwih =
      primitiv::functions::parameter<primitiv::Node>(wih);
    const primitiv::Node nwfh =
      primitiv::functions::parameter<primitiv::Node>(wfh);
    const primitiv::Node nwoh =
      primitiv::functions::parameter<primitiv::Node>(woh);
    const primitiv::Node nwjh =
      primitiv::functions::parameter<primitiv::Node>(wjh);

    const primitiv::Node nbi =
      primitiv::functions::parameter<primitiv::Node>(bi);
    const primitiv::Node nbf =
      primitiv::functions::parameter<primitiv::Node>(bf);
    const primitiv::Node nbo =
      primitiv::functions::parameter<primitiv::Node>(bo);
    const primitiv::Node nbj =
      primitiv::functions::parameter<primitiv::Node>(bj);

    const primitiv::Node i =
      primitiv::functions::sigmoid(
        primitiv::functions::matmul(nwix, x) +
        primitiv::functions::matmul(nwih, h) + nbi);
    const primitiv::Node f =
      primitiv::functions::sigmoid(
        primitiv::functions::matmul(nwfx, x) +
        primitiv::functions::matmul(nwfh, h) + nbf);
    const primitiv::Node o =
      primitiv::functions::sigmoid(
      primitiv::functions::matmul(nwox, x) +
      primitiv::functions::matmul(nwoh, h) + nbo);
    const primitiv::Node j =
      primitiv::functions::tanh(
      primitiv::functions::matmul(nwjx, x) +
      primitiv::functions::matmul(nwjh, h) + nbj);

    const primitiv::Node cc = f * c + i * j;
    const primitiv::Node hh = o * primitiv::functions::tanh(cc);

    const primitiv::Node t = primitiv::functions::zeros<primitiv::Node>({2});
    const primitiv::Node diff = hh - t;
    const primitiv::Node loss = diff * diff;
    const primitiv::Node sum_loss = primitiv::functions::batch::sum(
      primitiv::functions::sum(loss, 0));

    assert(45u == g.num_operators());

    const primitiv::Tensor loss_tensor = g.forward(loss);
    const primitiv::Tensor sum_loss_tensor = g.forward(sum_loss);

    const std::vector<float> expected_losses {
      5.7667205e-03, 2.8605087e-02, 1.4819370e-03, 3.0073307e-03 };

    const float expected_sum_loss = std::accumulate(
      std::begin(expected_losses), std::end(expected_losses), .0f);

    std::cout << expected_sum_loss << " = ";
    std::cout << sum_loss_tensor.to_float() << " = ";
    std::cout << sum_loss.to_float() << '\n';

    auto print = [](const std::string& name, const primitiv::Tensor& value) {
      std::cout << name << " : shape = " << value.shape().to_string()
        << ", values = [";
      const std::vector<float> data = value.to_vector();

      for (int i = 0; i < data.size(); ++i) {
        if (i > 0) {
          std::cout << ',';
        }
        std::cout << data[i];
      }
      std::cout << "]\n";
    };

#define PRINT(node) print(#node, g.forward(node))
  PRINT(x);
  PRINT(h);
  PRINT(c);
  PRINT(nwjx);
  PRINT(nwix);
  PRINT(nwfx);
  PRINT(nwox);
  PRINT(nwjx);
  PRINT(nwih);
  PRINT(nwfh);
  PRINT(nwoh);
  PRINT(nwjh);
  PRINT(nbi);
  PRINT(nbf);
  PRINT(nbo);
  PRINT(nbj);
  PRINT(i);
  PRINT(f);
  PRINT(o);
  PRINT(j);
  PRINT(cc);
  PRINT(hh);
  PRINT(t);
  PRINT(diff);
  PRINT(loss);
#undef PRINT
  }
  {
    primitiv::devices::Naive dev;
    primitiv::Device::set_default(dev);

    primitiv::Parameter wx(
      {8, 2},
      {.3, .1, .4, .1, .5, .9, .2, .6,
       .5, .3, .5, .8, .9, .7, .9, .3});
    primitiv::Parameter wh(
      {8, 2},
      {.2, .3, .8, .4, .6, .2, .6, .4,
       .3, .3, .8, .3, .2, .7, .9, .5});
    primitiv::Parameter b({8}, primitiv::initializers::Constant(0));

    primitiv::Graph g;
    primitiv::Graph::set_default(g);

    const primitiv::Node n_x =
      primitiv::functions::input<primitiv::Node>(
        primitiv::Shape({2}, 2), {2, -2, 0.5, -0.5});
    const primitiv::Node n_h =
      primitiv::functions::input<primitiv::Node>(
        primitiv::Shape({2}, 2), {-1, 1, -0.5, 0.5});
    const primitiv::Node n_c =
      primitiv::functions::zeros<primitiv::Node>({2});
    const primitiv::Node n_wx =
      primitiv::functions::parameter<primitiv::Node>(wx);
    const primitiv::Node n_wh =
      primitiv::functions::parameter<primitiv::Node>(wh);
    const primitiv::Node n_b =
      primitiv::functions::parameter<primitiv::Node>(b);

    const primitiv::Node u =
      primitiv::functions::matmul(n_wx, n_x) +
      primitiv::functions::matmul(n_wh, n_h) +
      n_b;
    const primitiv::Node i =
      primitiv::functions::sigmoid(
        primitiv::functions::slice(u, 0, 0, 2));
    const primitiv::Node f =
      primitiv::functions::sigmoid(
        primitiv::functions::slice(u, 0, 2, 4));
    const primitiv::Node o =
      primitiv::functions::sigmoid(
        primitiv::functions::slice(u, 0, 4, 6));
    const primitiv::Node j =
      primitiv::functions::tanh(
        primitiv::functions::slice(u, 0, 6, 8));
    const primitiv::Node cc = f * n_c + i * j;
    const primitiv::Node hh = o * primitiv::functions::tanh(cc);

    const primitiv::Node t = primitiv::functions::zeros<primitiv::Node>({2});
    const primitiv::Node diff = hh - t;
    const primitiv::Node loss = diff * diff;
    const primitiv::Node sum_loss =
      primitiv::functions::batch::sum(
        primitiv::functions::sum(loss, 0));

    
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
