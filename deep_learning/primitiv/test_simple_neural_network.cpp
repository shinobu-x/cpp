#include <primitiv/primitiv.h>

#include <iostream>

struct data {
  // Input data
  std::vector<float> input {
    1, 1,
    -1, 1,
    -1, -1,
    1, -1,
  };

  // Answer
  std::vector<float> ans {
    1,
    -1,
    1,
    -1,
  };
};

void doit() {

  primitiv::devices::Naive dev;
  primitiv::Device::set_default(dev);
  primitiv::Graph g;
  primitiv::Graph::set_default(g);

  data d;
  // A number of hidden layers
  const unsigned n = 8;
  /*x
   * Parameter objects for each data
   */
  primitiv::Parameter p1({1, n},
    primitiv::initializers::XavierUniform());
  primitiv::Parameter p2({},
    primitiv::initializers::Constant(0));
  primitiv::Parameter p3({n, 2},
    primitiv::initializers::XavierUniform());
  primitiv::Parameter p4({n},
    primitiv::initializers::Constant(0));

  primitiv::optimizers::SGD optimizer(0.5);
  optimizer.add(p1, p2, p3, p4);

  auto build_graph = [&] {
    auto x = primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2}, 4), d.input);
    auto w = primitiv::functions::parameter<primitiv::Node>(p1);
    auto b = primitiv::functions::parameter<primitiv::Node>(p2);
    auto u = primitiv::functions::parameter<primitiv::Node>(p3);
    auto c = primitiv::functions::parameter<primitiv::Node>(p4);
    auto h = primitiv::functions::tanh(
      primitiv::functions::matmul(u, x) * c);

    return primitiv::functions::tanh(
      primitiv::functions::matmul(w, h) + b);
  };

  auto calc_loss = [&](primitiv::Node y) {
    auto t = primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({}, 4), d.ans);
    auto diff = y - t;

    return primitiv::functions::batch::mean(diff * diff);
  };

  for (int epoch = 0; epoch < 20; ++epoch) {
    std::cout << epoch << '\n';

    g.clear();
    auto y = build_graph();

    for (auto v : y.to_vector()) {
      std::printf("%+.6f, ", v);
    }

    auto loss = calc_loss(y);
    std::printf("loss = %.6f", loss.to_vector());

    std::cout << '\n';

    optimizer.reset_gradients();
    loss.backward();
    optimizer.update();
  }
}

auto main() -> decltype(0) {
  doit();
  return 0;
}
