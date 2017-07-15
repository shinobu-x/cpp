/**
 * Cache Line:
 * -----------
 * Smallest unit that can be transferred between the main memory and cache. The
 * size of a cache line can be determined from the CPU specification, or direct-
 * ly retrieved from the processor by using the manufacturer's instruction set.
 *
 * [cpu#0] [cpu#1]     +-----------+
 *    |       |        | CL0 [ ]   | L1|D Cache|32k/core
 *    v       v        | ...       |
 *  [L1C]   [L2C] ---> | CL512 [ ] |
 *    |       |        +-----------+
 *     \_____/
 *        |            +-------------+
 *        v            | CL0 [ ]     | L2|D/I Cache|2MB Shared
 *  +-----------+      | ...         |
 *  |L2C(Shared)|      | CL32768 [ ] |
 *  +-----------+      +-------------+
 */

/**
 * Cache Coherency
 * ---------------
 * The protocol is to enforce data consistency among all the core's caches so t-
 * hat the system correctly processes valid data.
 * e.g.,
 * 1. CPU#1 updates Z in its own cache from 1 to 2.
 * 2. CPU#2 read Z in its own cache.
 * 3. Both CPU#1 and #2 have Z which value is 1.
 * 4. CPU#1 updates Z to 2.
 * 5. Employed with the < write-back > policy, CPU#1's cache doesn't need to im-
 *    mediately update the new value to the main memory.
 * 6. Z in the main memory and CPU#2's cache remains 1.
 * 7. CPU#1 must write 2 back to the main memory, and reload it to CPU#2's cache
 *    before CPU#2 start reading Z.
 *
 * a. CPU#1 update Z.
 * b. CPU#1 marks < Exclusive > to the cache line which Z resides.
 * c. CPU#1 allows load and store operations on the cache line.
 * d. CPU#2 needs to read Z.
 * e. CPU#2 marks the cache line as < Shared >
 * f. CPU#1 write 2 into cache line.
 * g. CPU#1 marks the cache line as < Modified >.
 * h. CPU#1 forces CPU#2 to < Invalidate > its cache line.
 * i. CPU#1 needs to backup Z with 2 to the main memory before CPU#2 can reload 
 *    2 to its cache line.
 */

/**
 * False Cache Line Sharing - False Sharing
 * ----------------------------------------
 * False sharing is a form of cache trashing caused by a mismatch between the m-
 * emory layout of write-shared data across processors and the reference pattern
 * to the data.
 * It could occur when 2 or more threads in parallel program are assinged to wo-
 * rk with different data elements in the same cache line.
 *
 * Thread#0 and Thread#1 update variables that adjacnet to each other located on
 * the same cache line.
 *
 * Although each thread modifies different variables, the cache line keeps bein-
 * g invalidated every iteration.
 *
 * 1. CPU#1 write a new value.
 * 2. CPU#1 make CPU#0's cache invalidated.
 * 3. CPU#1 causes the < write-back > to the main memory.
 * 4. CPU#0 updates value in its cache line.
 * 5. Invalidation will keep occuring between CPU#0's and #1's caches and the m-
 *    ain memory.
 *
 * As a result, the number of the main memory access increases considerably, and
 * causes great delays because of the high latency in data transfers between le-
 * vels of the memory hierarchy.
 *
 * +------------------+ +------------------+
 * | Thread#0 (CPU#0) | | Thread#1 (CPU#1) |
 * +------------------+ +------------------+
 * +------------------+ +------------------+
 * |[][x][][][][][][] | |[][][][][][y][][] | L1
 * +---^--------------+ +-----------^------+
 *     |                            |
 *      \                          /
 *       +--------+        +------+
 *                |        |
 * +--------------v--------v---------------+
 * |           [][x][][][][y][][]          | L2
 * +--------------------------------------+
 *                     ^
 *                     |
 *                     v 
 * +---------------------------------------+
 * |            [][][][][][][][]           | Memory
 * +---------------------------------------+
 */

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
