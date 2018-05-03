#ifndef MPIINITDEVICE_HPP
#define MPIINITDEVICE_HPP

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <vector>

#define NO_DEVICE -1

struct MPIGlobalState {
  int device = -1;
  cudaStream_t stream;
  bool initialized = false;
};

static MPIGlobalState global_state;

void mpiInitDevice(int device) {
  if (device < 0) {
    int mpi_status = MPI_Init(nullptr, nullptr);

    if (mpi_status != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Init failed.");
    }

    global_state.device = -1;
  } else {
    cudaError_t status = cudaSetDevice(device);

    if (status != cudaSuccess) {
      throw std::runtime_error("cudaSetDevice failed.");
    }

    cudaStreamCreate(&global_state.stream);

    int mpi_status = MPI_Init(nullptr, nullptr);

    if (mpi_status != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Init failed.");
    }

    global_state.device = device;
  }

  global_status.initialized = true;
}

float* mpiMalloc(float* ram, std::size_t size) {
  if (global_state.device < 0) {
    return (float*)malloc(size);
  }

  cudaMalloc((void**)&ram, sizeof(float)*size);
}

void mpiMdealloc(float* ram) {
  if (global_state.device < 0) {
    free(ram);
  } else {
    cudaFree(ram);
  }
}

void mpiMemcpy(float* dst, float* src, std::size_t size) {
  if (global_state.device < 0) {
    
#endif
