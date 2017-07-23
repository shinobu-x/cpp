#include <string>
#include <utility>
#include <vector>

void f1(std::vector<int> const& v) {
  std::vector<int> v_(v);
  v_.push_back(1);
}

void f2(std::vector<int>&&v) {
  v.push_back(1);
}

auto main() -> decltype(0)
{
  int&& a = 1;
  int b = 1;
  auto&& c = a;
  c = b;
//  int&& d = b;

  std::vector<int> v1;
  f1(v1);
  f2(std::move(v1));
}
