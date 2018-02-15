#include <iostream>
#include <random>
#include <tuple>

struct Data {
private:
  std::mt19937 rng_;
  std::normal_distribution<float> data_dist_, noise_dist_;

public:
  Data(float data_dist, float noise_dist) :
    rng_(std::random_device()()),
    data_dist_(0, data_dist),
    noise_dist_(0, noise_dist) {}

  std::tuple<float, float, float> operator()() {
    const float x1 = data_dist_(rng_);
    const float x2 = data_dist_(rng_);

    return std::make_tuple(
      x1 + noise_dist_(rng_),
      x2 + noise_dist_(rng_),
      x1 * x2 >= 0 ? 1 : -1);
  }
};

auto main() -> decltype(0) {
  float a = 2.3, b = 0.5;
  Data data(a, b);

  auto r = data();

  std::cout << std::get<0>(r) << '\n';
  std::cout << std::get<1>(r) << '\n';
  std::cout << std::get<2>(r) << '\n';

  return 0;
}
