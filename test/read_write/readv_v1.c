#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>

#include <fcntl.h>
#include <stdio.h>

/**
 * Scatter/gather I/O is a method of input and output where a single system call
 * writes to a vector of buffers from a single data stream, or alternatively re-
 * ads into a vector of buffers from a single data stream.
 */

int main() {
  int fd, i, max=3;
  char a[40], b[28], c[70];
  struct iovec iov[max];
  ssize_t n;

  // Create file descriptor
  fd = open("./file.txt", O_RDONLY);
  if (fd==-1) {
    perror("open: Failed");
    return 1;
  }

  // Complete iovec to read file
  iov[0].iov_len = sizeof(a);
  iov[1].iov_len = sizeof(b);
  iov[2].iov_len = sizeof(c);
  iov[0].iov_base = a;
  iov[1].iov_base = b;
  iov[2].iov_base = c;

  // Read file by a single system call!
  n = readv(fd, iov, max);
  if (n==-1) {
    perror("readv: Failed");
    return 1;
  }

  for (i=0; i<max; ++i)
    printf("%d: %s\n", i, (char*)iov[i].iov_base);

  if (close(fd)) {
    perror("close: Failed");
    return 1;
  } else
    unlink("./file.txt");

  return 0;
}
