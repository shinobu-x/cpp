#include <cstdlib>
#include <iostream>
#include <mpi.h>

/**
 * # Combines value from all ranks and distributes the result back to all ranks
 * MPI_Allreduce(
 *   const void* sendbuf,   # Starting address of sending buffer
 *   void* recvbuf,         # Starting address of receiving buffer
 *   int count,             # Number of elements in sending buffer
 *   MPI_Datatype datatype, # Data type of element of sending buffer
 *   MPI_Op op,             # Operation
 *   MPI_Comm comm          # Communicator
 * )
 */

auto main(int argc, char** argv) -> decltype(0) {
  int procs;
  int rank;
  int root;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int local;
  int global;

  if (rank != 0) {
    local = rank;
  }

  if (rank == 0) {
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }

  std::cout << "Rank: " << rank << " " << global << "\n";

  MPI_Finalize();

  return 0;
}
