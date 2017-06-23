#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#define SIZE 4096
#define JOBS 1000
#define LOOP 10
#define OUT std::cout <<
#define END << '\n';
int n_write=0, n_fsync=0;

template <typename T, T N>
T do_fsync() {
  char *b;
  int fd, r;
  char file[]="/var/lib/nova/instances/test.txt";
//  char file[]="/tmp/test.txt";
  std::cout << std::this_thread::get_id() << '\n';

  for (T i=0; i<LOOP; ++i) {
    if ((b = (char*)malloc(N)) == NULL)
      perror("Error: malloc");
    else if ((fd = creat(file, S_IWUSR)) < 0)
      perror("Error: creat");
    else {
      memset(b, 's', N);

      if ((r = write(fd, b, N)) == -1)
        perror("Error: write");
      else {
        n_write++;
        OUT "Write: " << n_write;
        OUT " / ";
        OUT "Fsync: " << n_fsync END;
        // std::cout << "Write " << r << " bytes\n";

        if (fsync(fd) != 0)
          perror("Error: fsync");
        else if ((r = write(fd, b, N)) == -1)
          perror("Error: write");
        else {
          n_fsync++;
          // std::cout << "Fsync: " << n_fsync << '\n';
          // std::cout << "Wrote " << r << " bytes\n";
         }
      }
    }

    close(fd);
    unlink(file);
  }
  return 0;
}

template <typename T, T N>
T doit() {
  std::vector<std::thread> v;


  for (T i=0; i<N; ++i)
    v.push_back(std::thread(do_fsync<T, SIZE>));

  for (auto& t : v)
    t.join();
}

auto main() -> int
{
  auto start = std::chrono::high_resolution_clock::now();

  doit<int, JOBS>();

  auto end = std::chrono::high_resolution_clock::now();

  OUT std::chrono::duration_cast<
    std::chrono::milliseconds>(end - start).count() END;
  return 0;
}
