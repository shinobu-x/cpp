#include <cstdlib>
#include <iostream>
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/timer/timer.hpp>

#define N 1 << 9

template <typename T>
void InitData(T* a, T* b) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      a[i][j] = 1.0;
      b[i][j] = 1.0;
    }
  }
}

template <typename T>
void Loop0(T* a, T* b) {
  boost::timer::cpu_timer t;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        a[i][j] = a[i][k] + b[k][j];
      }
    }
  }
  std::cout << t.format() << "\n";
}

template <typename T>
void Loop1(T* a, T* b) {
  boost::timer::cpu_timer t;
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < N; ++k) {
      for (int j = 0; j < N; ++j) {
        a[i][j] = a[i][k] + b[k][j];
      }
    }
  }
  std::cout << t.format() << "\n";
}

template <typename T>
void Loop2(T* a, T* b) {
  boost::timer::cpu_timer t;
  int BLOCK = 1 << 3;
  for (int ii = 0; ii < N; ii += BLOCK) {
    for (int kk = 0; kk < N; kk += BLOCK) {
      for (int jj = 0; jj < N; jj += BLOCK) {
        for (int i = ii; i < ii + BLOCK && i < N; ++i) {
          for (int k = kk; k < kk + BLOCK && k < N; ++k) {
            for (int j = jj; j < jj + BLOCK && j < N; ++j) {
              a[i][j] = a[i][k] + b[k][j];
            }
          }
        }
      }
    }
  }
  std::cout << t.format() << "\n";
}

template <typename T>
void DoIt() {
  typedef T value_type;
  value_type a[N][N];
  value_type b[N][N];

  InitData(a, b);
  Loop0(a, b);

  boost::this_thread::sleep_for(boost::chrono::seconds(3));

  InitData(a, b);
  Loop1(a, b);

  boost::this_thread::sleep_for(boost::chrono::seconds(3));

  InitData(a, b);
  Loop1(a, b);
}

auto main() -> decltype(0) {
  DoIt<float>();
  return 0;
}
