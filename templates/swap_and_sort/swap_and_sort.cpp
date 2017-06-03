#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
class sorted_vector {
public:
  void swap(srted_vector<T>& that) {
    data_.swap(that.data_);
  }

  void swap(std::vector<T>& that) {
    data_.swap(that);
    std::sort(data_.begin(), data_.end());
  }

private:
  std::vector<T> data_;
};

template <typename T>
T doit() {
  sorted_vector<T> x;
  std::vector<T> y;

  std::random_device rd;

  for (int i = 0; i < 10; ++) {
    std::cout << rd() << '\n';
  }

auto main() -> int
{
  return 0;
}
