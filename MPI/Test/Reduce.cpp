#include <iostream>
#include <mpi.h>

/**
 * MPI_Reduce(
 *   const void* sendbuf,   # Address of send buffer
 *   void* recvbuf,         # Address of receive buffer
 *   int count,             # Number of element in send buffer
 *   MPI_Datatype datatype, # Data type of elements of send buffer
 *   MPI_Op op,             # Reduce operation
 *   int root,              # Rank of root process
 *   MPI_Comm comm          # Communicator
 * )
 */

auto main(int argc, char** argv) -> decltype(0) {
  int procs;
  int rank;
  int data = 1;
  int result;
  int reducer = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Reduce(&data, &result, 1, MPI_INT, MPI_SUM, reducer, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Rank: " << rank << " | Result: " << result << "\n";
  } else {
    std::cout << "Rank: " << rank << " | Sent: " << data << "\n";
  }

  MPI_Finalize();

  return 0;
}
