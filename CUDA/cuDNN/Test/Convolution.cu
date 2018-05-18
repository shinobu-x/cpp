#include <cuda.h>
#include <cudnn.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <CUDA/HPP/Utils.hpp>

/**
 * # Initialization
 * 1.  cudnnCreate
 * # Input
 * 2.  cudnnCreateTensorDescriptor
 * 3.  cudnnSetTensor4dDescriptor
 * # Filter
 * 4.  cudnnCreateFilterDescriptor
 * 5.  cudnnSetFilter4dDescriptor
 * # Convolution
 * 6.  cudnnCreateConvolutionDescriptor
 * 7.  cudnnSetConvolution2dDescriptor
 * # Output
 * 8.  cudnnGetConvolution2dForwardOutputDim
 * 9.  cudnnCreateTensorDescriptor
 * 10. cudnnSetTensor4dDescriptor
 * # Algorithm
 * 11. cudnnGetConvolutionForwardAlgorithm
 * 12. cudnnGetConvolutionBackwardAlgorithm
 * 13. cudnnGetConvolutionBackwardFilterAlgorithm
 * # Workspace
 * 14. cudnnGetConvolutionForwardWorkspaceSize
 * 15. cudnnGetConvolutionBackwardDataWorkspaceSize
 * 16. cudnnGetConvolutionBackwardFilterWorkspaceSize
 * # Forward propargation
 * 17. cudnnConvolutionForward
 * # Backward propargation
 * 18. cudnnConvolutionBackwardData
 * 19. cudnnConvolutionBackwardFilter
 */

__global__
void InitData(float* px, float a, float b = -1.0) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (b != -1.0) {
    px[tid] = a;
  } else {
    px[tid] = tid + b;
  }
}

auto main() -> decltype(0) {

  const int xW = 5;
  const int xH = 5;
  const int xC = 1;
  const int xN = 1;

  const int wW = 2;
  const int wH = 2;
  const int wC = 1;
  const int wK = 1;

  const int padW = 0;
  const int padH = 0;

  const int strideW = 1;
  const int strideH = 1;

  const int dilationW = 1;
  const int dilationH = 1;

  const int xBias = 1;
  const int wBias = 1;

  // Initialization (creates context)
  cudnnHandle_t ctx;
  cudnnCheck(cudnnCreate(&ctx));

  // Input
  cudnnTensorDescriptor_t xDesc;
  cudnnCheck(cudnnCreateTensorDescriptor(&xDesc));
  cudnnCheck(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW,
    CUDNN_DATA_FLOAT, xN, xC, xH, xW));

  // Filter
  cudnnFilterDescriptor_t wDesc;
  cudnnCheck(cudnnCreateFilterDescriptor(&wDesc));
  cudnnCheck(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT,
    CUDNN_TENSOR_NCHW, wK, wC, wH, wW));

  // Convolution
  cudnnConvolutionDescriptor_t cDesc;
  cudnnCheck(cudnnCreateConvolutionDescriptor(&cDesc));
  cudnnCheck(cudnnSetConvolution2dDescriptor(cDesc, padH, padW,
    strideH, strideW, dilationH, dilationW, CUDNN_CONVOLUTION,
    CUDNN_DATA_FLOAT));

  // Output
  int yN, yC, yH, yW;
  cudnnCheck(cudnnGetConvolution2dForwardOutputDim(cDesc, xDesc, wDesc,
    &yN, &yC, &yH, &yW));

  cudnnTensorDescriptor_t yDesc;
  cudnnCheck(cudnnCreateTensorDescriptor(&yDesc));
  cudnnCheck(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW,
    CUDNN_DATA_FLOAT, yN, yC, yH, yW));

  // Algorithm
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdXAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdWAlgo;

  cudnnCheck(cudnnGetConvolutionForwardAlgorithm(ctx, xDesc, wDesc, cDesc,
    yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwdAlgo));
  cudnnCheck(cudnnGetConvolutionBackwardDataAlgorithm(ctx, wDesc, yDesc,
    cDesc, xDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwdXAlgo));
  cudnnCheck(cudnnGetConvolutionBackwardFilterAlgorithm(ctx, xDesc, yDesc,
    cDesc, wDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwdWAlgo));

  // Workspace
  std::size_t fwdWSsize;
  std::size_t bwdXWSsize;
  std::size_t bwdWWSsize;
  cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(ctx, xDesc, wDesc, cDesc,
    yDesc, fwdAlgo, &fwdWSsize));
  cudnnCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(ctx, wDesc, yDesc,
    cDesc, xDesc, bwdXAlgo, &bwdXWSsize));
  cudnnCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(ctx, xDesc, yDesc,
    cDesc, wDesc, bwdWAlgo, &bwdWWSsize));

  typedef float value_type;
  std::size_t value_size = sizeof(value_type);
  auto X = MemAlloc<value_type>(xN * xC * xH * xW * value_size);
  auto W = MemAlloc<value_type>(wK * wC * wH * wW * value_size);
  auto Y = MemAlloc<value_type>(yN * yC * yH * yW * value_size);
  auto dX = MemAlloc<value_type>(xN * xC * xH * xW * value_size);
  auto dW = MemAlloc<value_type>(wK * wC * wH * wW * value_size);
  auto dY = MemAlloc<value_type>(yN * yC * yH * yW * value_size);

  auto fwdWS = MemAlloc(fwdWSsize);
  auto bwdXWS = MemAlloc(bwdXWSsize);
  auto bwdWWS = MemAlloc(bwdWWSsize);

  InitData<<<xW * xH, xN * xC>>>(X.get(), xBias);
  InitData<<<wW * wH, wK * wC>>>(W.get(), wBias);
  InitData<<<xW * xH, xN * xC>>>(dX.get(), 0.0, 1.0);
  InitData<<<yW * yH, yN * yC>>>(dY.get(), 0.0, 0.0);
  InitData<<<wW * wH, wK * wC>>>(dW.get(), 0.0, 0.0);

  value_type alpha = 1.f;
  value_type beta = 0.f;
  cudnnCheck(cudnnConvolutionForward(ctx, &alpha, xDesc, X.get(), wDesc,
    W.get(), cDesc, fwdAlgo, fwdWS.get(), fwdWSsize, &beta, yDesc, Y.get()));

  alpha = 1.f;
  beta = 1.f;
  cudnnCheck(cudnnConvolutionBackwardData(ctx, &alpha, wDesc, W.get(), yDesc,
    dY.get(), cDesc, bwdXAlgo, bwdXWS.get(), bwdXWSsize, &beta, xDesc,
    dX.get()));

  cudnnCheck(cudnnConvolutionBackwardFilter(ctx, &alpha, xDesc, X.get(), yDesc,
    dY.get(), cDesc, bwdWAlgo, bwdWWS.get(), bwdWWSsize, &beta, wDesc,
    dW.get()));

  cudnnCheck(cudnnDestroyTensorDescriptor(xDesc));
  cudnnCheck(cudnnDestroyTensorDescriptor(yDesc));
  cudnnCheck(cudnnDestroyFilterDescriptor(wDesc));
  cudnnCheck(cudnnDestroyConvolutionDescriptor(cDesc));
  cudnnCheck(cudnnDestroy(ctx));

  return 0;
}
