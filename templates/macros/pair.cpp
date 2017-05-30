#include <iostream>
#include <utility>

// #define id first
// #define value second

// #define id(p) p.first
// #define value(p) p.second
inline int& id(std::pair<int, double>& p) {
  return p.first;
}

inline double& value(std::pair<int, double>& p) {
  return p.second;
}

// Global
typedef std::pair<int, double> id_value;
int id_value::*ID = &id_value::first;
double id_value::*VALUE = &id_value::second;

template <typename T, typename U>
void doit() {
  // Use them
  std::pair<T, U> p;
  p.*ID = -5;
  p.*VALUE = 1.23;
  std::cout << p.*ID << '\n';
  std::cout << p.*VALUE << '\n';
}

auto main() -> int
{
  doit<int, double>();
  return 0;
}
