#include <cstdlib>
#include <mpi.h>
#include <iostream>

/**
 * # Sendss data from one process to all other processes in a communicator
 * MPI_Scatter(
 *   const void* sendbuf,   # Address of sending buffer
 *   int sendcount,         # Number of elements sent to each process
 *   MPI_Datatype sendtype, # Data type of sending buffer elements
 *   void* recvbuf,         # Address of receiving buffer
 *   int recvcount,         # Number of elements in receiving buffer
 *   MPI_Datatype recvtype, # Data type of receiving buffer elements
 *   int root,              # Rank of sending process
 *   MPI_Comm comm          # Communicator
 * )
 */

auto main(int argc, char** argv) -> decltype(0) {
  int procs;
  int rank;
  int root = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int a;
  int data[sizeof(int) * procs];

  if (rank == 0) {
    for (int i = 0; i < procs; ++i) {
      data[i] = i;
    }
  }

  MPI_Scatter(&data, 1, MPI_INT, &a, 1, MPI_INT, root, MPI_COMM_WORLD);

  std::cout << "Rank: " << rank << " | Data: " << a << "\n";

  MPI_Finalize();

  return 0;
}
