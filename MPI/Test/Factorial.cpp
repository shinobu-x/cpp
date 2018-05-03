#include <iostream>
#include <mpi.h>

auto main(int argc, char** argv) -> decltype(0) {
  int my_rank;
  int size;
  int local_result;
  int result = 0;
  MPI_Status status;
  const int tag = 0;
  const int load = 1;
  const int destination = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (my_rank == 0) {
    for (int i = 1; i < size; ++i) {
      /**
       * Input:
       *  buf: Initial address of receive buffer.
       *  count: Maximum number of elements to receive.
       *  datatype: Data type of each receive buffer entry.
       *  source: Rank of source.
       *  tag: Message tag.
       *  comm: Communicator.
       *  status: Status object.
       */
      MPI_Recv(&local_result, load, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
      std::cout << "Proc: " << local_result << "\n";
      result += local_result;
    }
    std::cout << "Total: " << result << "\n";
  } else {
    /**
     * Input:
     *  buf: Initial address of send buffer.
     *  count: Number of elements to send.
     *  datatype: Data type of each send buffer element.
     *  dest: Rank of destination.
     *  tag: Message tag.
     *  comm: Communicator.
     */
    MPI_Send(&my_rank, load, MPI_INT, destination, tag, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
