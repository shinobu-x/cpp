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
struct Loop0 {
  void operator()() {
    T a[N][N];
    T b[N][N];
    ::InitData(a, b);
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
};

template <typename T>
struct Loop1 {
  void operator()() {
    T a[N][N];
    T b[N][N];
    ::InitData(a, b);
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
};

template <typename T>
struct Loop2 {
  void operator()() {
    T a[N][N];
    T b[N][N];
    ::InitData(a, b);
    boost::timer::cpu_timer t;
    int BLOCK = 1 << 6;
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
};

template <typename T>
void DoIt() {
  typedef T value_type;

  Loop0<value_type> L0;
  boost::thread l0(L0);
  l0.join();

  boost::this_thread::sleep_for(boost::chrono::seconds(3));

  Loop1<value_type> L1;
  boost::thread l1(L1);
  l1.join();

  boost::this_thread::sleep_for(boost::chrono::seconds(3));

  Loop2<value_type> L2;
  boost::thread l2(L2);
  l2.join();
}

auto main() -> decltype(0) {
  DoIt<float>();
  return 0;
}
