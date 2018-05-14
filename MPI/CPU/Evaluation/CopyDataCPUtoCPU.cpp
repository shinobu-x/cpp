#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <iostream>

void InitData(char* a, char* b, std::size_t s) {
  memset(a, '0', s);
  memset(b, '0', s);
}

auto main(int argc, char** argv) -> decltype(0) {

  typedef char value_type;
  typedef char* ptr_type;
  int N = 1<<22;
  std::size_t NBytes = N * sizeof(value_type);
  ptr_type s_buf = (ptr_type)malloc(NBytes);
  ptr_type r_buf = (ptr_type)malloc(NBytes);
  int rank;
  int size;
  char procs[MPI_MAX_PROCESSOR_NAME];
  int proc;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Get_processor_name(procs, &proc);

  MPI_Status status;
  MPI_Request s_req;
  MPI_Request r_req;

  auto other = (rank == 1 ? 0 : 1);

  if (size != 2) {
    if (rank == 0) {
      std::cout << "Number of processors: " << size << "\n";
      MPI_Finalize();
      return 1;
    }
  }

  double start;
  double end;

  for (int i = 1024; i <= 1<<22; i *= 4) {
    InitData(s_buf, r_buf, i);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      start = MPI_Wtime();

      for (int j = 0; j < 1<<7; ++j) {
        MPI_Irecv(r_buf, i, MPI_CHAR, other, 10, MPI_COMM_WORLD, &r_req);
        MPI_Isend(s_buf, i, MPI_CHAR, other, 100, MPI_COMM_WORLD, &s_req);
        MPI_Waitall(1, &s_req, &status);
        MPI_Waitall(1, &r_req, &status);
      }

      end = MPI_Wtime();
    } else {
      for (int j = 0; j < 1<<7; ++j) {
        MPI_Irecv(r_buf, i, MPI_CHAR, other, 100, MPI_COMM_WORLD, &r_req);
        MPI_Isend(s_buf, i, MPI_CHAR, other, 10, MPI_COMM_WORLD, &s_req);
        MPI_Waitall(1, &s_req, &status);
        MPI_Waitall(1, &r_req, &status);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      auto lat = (end - start) * 1e6 / (2.0 * (1<<6));
      auto perf = (value_type)i / (value_type)lat;
      auto a = (i >= 1<<10 * 1<<10) ? i / 1<<10 / 1<<10 : i / 1<<10;
      auto b = (i >= 1<<10 * 1<<10) ? "MB" : "KB";
    }
  }

  free(s_buf);
  free(r_buf);

  MPI_Finalize();

  return 0;
}
