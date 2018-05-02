#include <cmath>
#include <cstdlib>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <CUDA/HPP/CalElapsedTime.hpp>

typedef float value_type;
int N = std::pow(2, 10);
std::size_t NBytes = N * N * sizeof(value_type);

value_type* a;
value_type* b;
value_type* c;
int i, j, k;

void InitRowMajor() {
  a = (value_type*)malloc(NBytes);
  b = (value_type*)malloc(NBytes);
  c = (value_type*)malloc(NBytes);

  for (i = 0; i < N; ++i) {
    for (k = 0; k < N; ++k) {
      a[i * N + k] = (float)(i + 1) * 0.1f;
    }
  }

  for (k = 0; k < N; ++k) {
    for (j = 0; j < N; ++j) {
      b[k * N + j] = (float)(j + 1) * 0.1f;
    }
  }

  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      c[i * N + j] = 0.0f;
    }
  }
}

void InitColumnMajor() {
  a = (value_type*)malloc(NBytes);
  b = (value_type*)malloc(NBytes);
  c = (value_type*)malloc(NBytes);

  for (k = 0; k < N; ++k) {
   for (i = 0; i < N; ++i) {
     a[k * N + i] = (float)(i + 1) * 0.1f;
   }
  }

  for (j = 0; j < N; ++j) {
    for (k = 0; k < N; ++k) {
      b[j * N + k] = (float)(k + 1) * 0.1f;
    }
  }

  for (j = 0; j < N; ++j) {
    for (i = 0; i < N; ++i) {
      c[j * N + i] = 0.0f;
    }
  }
}

void MatmulRowMajor() {
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      for (k = 0; k < N; ++k) {
        c[i * N + j] += a[i * N + k] * b[k * N + j];
      }
    }
  }
}

void MatmulColumnMajor() {
  for (j = 0; j < N; ++j) {
    for (i = 0; i < N; ++i) {
      for (k = 0; k < N; ++k) {
        c[j * N + i] += a[k * N + i] + b[j * N + k];
      }
    }
  }
}

auto main() -> decltype(0) {
  InitRowMajor();
  std::cout << CalElapsedTime<>::Execution(MatmulRowMajor) << "\n";

  boost::this_thread::sleep_for(boost::chrono::seconds(5));

  InitColumnMajor();
  std::cout << CalElapsedTime<>::Execution(MatmulColumnMajor) << "\n";

  free(a);
  free(b);
  free(c);

  return 0;
} 
