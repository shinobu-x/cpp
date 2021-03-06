#ifndef SPAWNKERNEL_HPP
#define SPAWNKERNEL_HPP

#include <cstdlib>
#include <cuda_runtime.h>
#include <boost/thread.hpp>
#include <iostream>

#include <CUDA/HPP/InitData.hpp>
#include <CUDA/HPP/cudaSetupDevice.hpp>
#include <CUDA/HPP/sumArraysOnDevice.hpp>

template <typename ValueType = float>
struct SpawnKernel {

  void operator()(int thread_num = 0) {
    std::cout << boost::this_thread::get_id() << '\n';

    typedef ValueType value_type;
    if (thread_num == 0) {
      int threads = 1 << 24;
    } else {
      int threads = thread_num;
    }
    std::size_t size = threads * sizeof(value_type);

    // Addresses reservations for Host
    value_type* h_a;
    value_type* h_b;
    value_type* h_c;

    h_a = (value_type*)malloc(size);
    h_b = (value_type*)malloc(size);
    h_c = (value_type*)malloc(size);

    // Set initial data
    InitData(h_a, threads);
    InitData(h_b, threads);

    // Addresses reservations for Device
    value_type* d_a;
    value_type* d_b;
    value_type* d_c;

    cudaMalloc((value_type**)&d_a, size);
    cudaMalloc((value_type**)&d_b, size);
    cudaMalloc((value_type**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << *h_c << '\n';

    dim3 block(threads);
    dim3 grid((threads + block.x - 1) / block.x);
    sumArraysOnDevice<<<grid, block>>>(d_a, d_b, d_c, threads);
    cudaDeviceSynchronize();

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

#endif
