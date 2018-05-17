#include <cstdlib>
#include <iostream>
#include <mpi.h>

/**
 * # Broadcasts a message to all other processes of the communicator
 * MPI_Bcast(
 *   void* buffer,          # Starting address of buffer
 *   int count,             # Number of entries in buffer
 *   MPI_Datatype datatype, # Data type of buffer
 *   int root,              # Rank of broadcast root
 *   MPI_Comm               # Communicator
 * )
 */

auto main(int argc, char** argv) -> decltype(0) {
  int procs;
  int rank;
  int source = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int data[sizeof(int) * procs];
  for (int i = 0; i < procs; ++i) {
    data[i] = i + 1;
  }

  MPI_Bcast(&data, procs, MPI_INT, source, MPI_COMM_WORLD);

  if (rank != source) {
    for (int i = 0; i < procs; ++i) {
      std::cout << "Rank: " << rank << " | Data: " << data[i] << "\n";
    }
  }

  MPI_Finalize();

  return 0;
}
