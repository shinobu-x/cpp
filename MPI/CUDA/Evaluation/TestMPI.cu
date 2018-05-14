#include <mpi.h>

__global__
void Kernel() {
}

auto main(int argc, char** argv) -> decltype(0) {
  MPI_Init(&argc, &argv);
  int rank;
  int size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_rank(MPI_COMM_WORLD, &size);

  dim3 Block(1024);
  dim3 Grid((1024 * Block.x - 1) / Block.x);
  Kernel<<<Grid, Block>>>();

  MPI_Finalize();

  return 0;
}
