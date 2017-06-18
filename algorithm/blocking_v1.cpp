/**
 * Blocking is a general optimization technique for increasing the effectivenes-
 * s of a memory hierarchy. By reusing data in the faster level of the hierarch-
 * y, it cuts down the average access latency. It also reduces the number of re-
 * ferences made to slower level of hierarchy. Blocking is thus superior to opt-
 * imization such as prefetching, which hides the latency but does not reduce t-
 * he memory bandwidth requirement. This reduction is especially important for 
 * multiprocessors since memory bandwidth is often the bottleneck of the system.
 * Blocking has been shown to be usefull for many algorithms in linear algebra.
 */
#include <iostream>

#define BLOCKSIZE 1

template <typename T, typename U>
void do_blocking(T n, T p, T q, T r, U* a[], U* b[], U* c[]) {
  for (T i=p; i<p+BLOCKSIZE; ++i)
    for (T j=q; j<q+BLOCKSIZE; ++j) {
      U cij = *c[i+j*n];
      for (T k=r; k<r+BLOCKSIZE; ++k)
         cij = *a[i+k*n] * *b[k+j*n];
      *c[i+j*n] = cij;
    }
}

template <typename T, typename U>
void blocking(T n, U* a[], U* b[], U* c[]) {
  for (T p=0; p<n; p+=BLOCKSIZE)
    for (T q=0; q<n; q+=BLOCKSIZE)
      for (T r=0; r<n; r+=BLOCKSIZE)
        do_blocking<T, U>(n, p, q, r, a, b, c);
}

template <typename T, T N, typename U>
void doit() {
  U* a[] = {};
  U* b[] = {};
  U* c[] = {};
  blocking<T, U>(N, a, b, c);
}

auto main() -> int
{
  doit<int, 3, double>();
  return 0;
}
