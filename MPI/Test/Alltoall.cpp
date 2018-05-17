#include <cstdlib>
#include <iostream>
#include <mpi.h>

/**
 * # Send data from all to all ranks
 * MPI_Alltoall(
 *   const void* sendbuf,   # Starting address of sending buffer
 *   int sendcount,         # Number of elements to send to each rank
 *   MPI_Datatype sendtype, # Data type of sending buffer elements
 *   void* recvbuf,         # Address of receiving buffer
 *   int recvcount,         # Number of elements received from any ranks
 *   MPI_Datatype recvtype, # Data type of receiving buffer elements
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

  int a[sizeof(int) * procs];
  for (int i = 0; i < procs; ++i) {
    a[i] = i + 1;
  }

  int data[sizeof(int) * procs];
  MPI_Alltoall(&a, 1, MPI_INT, &data, 1, MPI_INT, MPI_COMM_WORLD);

  std::cout << "Rank: " << rank << "\n";
  for (int i = 0; i < procs; ++i) {
    std::cout << data[i] << "\n";
  }

  MPI_Finalize();

  return 0;
}
