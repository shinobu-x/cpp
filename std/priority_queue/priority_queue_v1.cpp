#include <iostream>
#include <queue>
#include <random>

template <typename T>
void doit() {
  std::random_device rd;
  std::priority_queue<T> q;
  T v;

  for (int i=0; i<10; ++i) {
    v = rd();
    q.push(v);
  }    

  while (!q.empty()) {
    std::cout << q.top() << '\n';
    q.pop();
  }
}

auto main() -> decltype(0) 
{
  doit<unsigned int>();
  return 0;
}
