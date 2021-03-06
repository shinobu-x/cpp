#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
class sorted_vector {
public:
  void swap(sorted_vector<T>& that) {
    data_.swap(that.data_);
  }

  void swap(std::vector<T>& that) {
    data_.swap(that);
    std::sort(data_.begin(), data_.end());
  }

  void load_data() {
    std::random_device rd;
    for (int i = 0; i < 10; ++i) {
      data_.push_back(rd());
    }
  }

private:
  std::vector<T> data_;
};

template <typename T>
T doit() {
  sorted_vector<T> x;
  std::vector<T> y;

  std::random_device rd;

  for (int i = 0; i < 10; ++i) {
    y.push_back(rd());
  }

  x.load_data();
  x.swap(y);

}
typedef unsigned int type;
auto main() -> int
{
  doit<type>();
  return 0;
}
