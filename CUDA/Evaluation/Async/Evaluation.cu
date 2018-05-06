// nvprof --print-gpu-trace ./a.out
#define BOOST_THREAD_VERSION 4
#include <boost/thread/future.hpp>
#include <boost/thread.hpp>
#include <cassert>
#include <cstdlib>
#include <CUDA/HPP/InitData.hpp>
#include <CUDA/HPP/sumArraysOnDevice.hpp>

typedef float value_type;
const int N = 8;
std::size_t NBytes = N * sizeof(value_type);
// cudaStream_t stream = nullptr;

void SpawnKernel(cudaStream_t stream = nullptr) {
  value_type* h_a;
  value_type* h_b;
  value_type* h_c;
  h_a = (value_type*)malloc(NBytes);
  h_b = (value_type*)malloc(NBytes);
  h_c = (value_type*)malloc(NBytes);

  InitData(h_a, N);
  InitData(h_b, N);

  value_type* d_a;
  value_type* d_b;
  value_type* d_c;
  cudaMalloc((value_type**)&d_a, NBytes);
  cudaMalloc((value_type**)&d_b, NBytes);
  cudaMalloc((value_type**)&d_c, NBytes);

  cudaMemcpy(d_a, h_a, NBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, NBytes, cudaMemcpyHostToDevice);

  dim3 Block(N);
  dim3 Grid((N + Block.x - 1) / Block.x);
  if (stream) {
    sumArraysOnDevice<<<Grid, Block, 0, stream>>>(
      d_a, d_b, d_c, N);
    cudaMemcpyAsync(h_c, d_c, NBytes, cudaMemcpyDeviceToHost, stream);
  } else {
    sumArraysOnDevice<<<Grid, Block>>>(d_a, d_b, d_c, N);
  }
  cudaDeviceSynchronize();

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void DoStream() {
  cudaStream_t s[N];
  value_type* Data[N];

  for (int i = 0; i < N; ++i) {
    cudaStreamCreate(&s[i]);
    cudaMalloc(&Data[i], NBytes);
    SpawnKernel(s[i]);
  }
}

void DoFuture() {
  boost::future<void> f[N];

  for (auto& f_ : f) {
    boost::packaged_task<void(cudaStream_t)> t(SpawnKernel);
    f_ = t.get_future();
    boost::thread(boost::move(t), nullptr).detach();
  }

  for (auto& f_ : f) {
    f_.get();
    assert(f_.is_ready());
    assert(f_.has_value());
    assert(!f_.has_exception());
    assert(f_.get_state() == boost::future_state::ready);
  }
}

void DoAsync() {
  boost::future<void> f[N];

  for (int i = 0; i < N; ++i) {
    f[i] = boost::async(boost::launch::async, []() mutable {
      SpawnKernel(nullptr);
    });
  }

  for (auto& f_ : f) {
    f_.get();
    assert(f_.is_ready());
    assert(f_.has_value());
    assert(!f_.has_exception());
    assert(f_.get_state() == boost::future_state::ready);
  }

}
auto main() -> decltype(0) {
//  DoStream();
//  DoFuture();
  DoAsync();
  return 0;
}
