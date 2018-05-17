#include <cstdlib>
#include <iostream>
#include <mpi.h>

/**
 * # Gathers data from all ranks and distributed the combined data to all ranks
 * MPI_Allgather(
 *   const void* sendbuf,   # Starting address of sending buffer
 *   int sendcount,         # Number of elements in sending buffer
 *   MPI_Datatype sendtype, # Data type of sending buffer elements
 *   void* recvbuf,         # Address of receiving buffer
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

  int a = rank;
  int data[sizeof(int) * procs];

  MPI_Allgather(&a, 1, MPI_INT, &data, 1, MPI_INT, MPI_COMM_WORLD);

  for (int i = 0; i < procs; ++i) {
    std::cout << "Data: " << data[i] << "\n";
  }

  MPI_Finalize();

  return 0;
}
