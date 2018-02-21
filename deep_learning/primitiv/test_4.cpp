#include <primitiv/primitiv.h>

#include <iostream>

auto main() -> decltype(0) {
  primitiv::devices::Naive d;
  primitiv::Device::set_default(d);
  primitiv::Graph g;
  primitiv::Graph::set_default(g);

  primitiv::Parameter w1({8, 2}, primitiv::initializers::XavierUniform());
  primitiv::Parameter w2({1, 8}, primitiv::initializers::XavierUniform());
  primitiv::Parameter b1({8}, primitiv::initializers::Constant(0));
  primitiv::Parameter b2({}, primitiv::initializers::Constant(0));

  primitiv::optimizers::SGD optimizer(0.1);
  optimizer.add(w1, b1, w2, b2);

  const std::vector<float> input_data {
    1,  1,
    1, -1,
   -1,  1,
   -1, -1,
  };

  const std::vector<float> output_data {
    1,
   -1,
   -1,
    1,
  };

  for (int i = 0; i < 10; ++i) {
    g.clear();

    // Computation graphs
    const primitiv::Node x = primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({2}, 4), input_data);
    const primitiv::Node nw1 =
      primitiv::functions::parameter<primitiv::Node>(w1);
    const primitiv::Node nb1 =
      primitiv::functions::parameter<primitiv::Node>(b1);
    const primitiv::Node nw2 =
      primitiv::functions::parameter<primitiv::Node>(w2);
    const primitiv::Node nb2 =
      primitiv::functions::parameter<primitiv::Node>(b2);
    const primitiv::Node h = primitiv::functions::tanh(
      primitiv::functions::matmul(nw1, x) + nb1);
    const primitiv::Node y = primitiv::functions::matmul(nw2, h) + nb2;

    const std::vector<float> v1 = y.to_vector();
    std::cout << "Epoch: " << i << ": \n";
    for (int j = 0; j < 4; ++j) {
      std::cout << " [ " << j << " ] " << v1[j] << '\n';
    }

    // For mean squared loss
    const primitiv::Node t = primitiv::functions::input<primitiv::Node>(
      primitiv::Shape({}, 4), output_data);
    const primitiv::Node diff = t - y;
    const primitiv::Node loss = primitiv::functions::batch::mean(diff * diff);

    // Loss
    const float v2 = loss.to_float();
    std::cout << "Loss: " << v2 << '\n';

    optimizer.reset_gradients();
    loss.backward();
    optimizer.update();
  }

  return 0;
}
