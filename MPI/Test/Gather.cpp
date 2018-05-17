#include <cstdlib>
#include <iostream>
#include <mpi.h>

/**
 * # Gathers together values from a group of processes
 * MPI_Gather(
 *   const void* sendbuf,   # Starting address of send buffer
 *   int sendcount,         # Number of elements in send buffer
 *   MPI_Datatype sendtype, # Data type of send buffer elements
 *   void* recvbuf,         # Address of receiving buffer
 *   int recvcount,         # Number of elements for any single receive
 *   MPI_Datatype recvtype, # Data type of recv buffer elements
 *   int root,              # Rank of receiving process
 *   MPI_Comm comm          # Communicator
 * )
 */

auto main(int argc, char** argv) -> decltype(0) {
  int procs;
  int rank;
  int a;
  int root = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int data[sizeof(int) * procs];
  a = rank;

  MPI_Gather(&a, 1, MPI_INT, data, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < procs; ++i) {
      std::cout << data[i] << "\n";
    }
  }

  MPI_Finalize();

  return 0;
}
