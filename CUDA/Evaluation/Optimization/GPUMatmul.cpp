#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <CUDA/HPP/CalElapsedTime.hpp>

typedef float value_type;
int N = std::pow(2, 10);
std::size_t NBytes = N * N * sizeof(value_type);


void Init()
