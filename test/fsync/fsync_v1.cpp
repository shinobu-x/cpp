#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>

#define M_STRING_LENGTH 250000

int main() {
  char *buffer;
  int fd, ret;
  char fn[]="/tmp/fsync.file";

  if ((buffer = (char*)malloc(M_STRING_LENGTH)) == NULL)
    perror("malloc error");
  else if ((fd = creat(fn, S_IWUSR)) < 0)
    perror("create error");
  else {
    // Write s*M_STRING_LENGTH to buffer
    memset(buffer, 's', M_STRING_LENGTH);

    if ((ret = write(fd, buffer, M_STRING_LENGTH)) == -1)
      perror("write error");
    else {
      std::cout << "Write " << ret << " bytes\n";

      if (fsync(fd) != 0)
        perror("fsync error");
      // Write buffered data to fd
      else if ((ret = write(fd, buffer, M_STRING_LENGTH)) == -1)
        perror("write error");
      else
        std::cout << "Wrote " << ret << " bytes\n";
    }
    close(fd);
    unlink(fn);
  }
  return 0;
}
