#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main()
{
  int fd, i, max = 3;
  struct iovec iov[max];
  ssize_t n;

  char *buf[] = {
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "ccccccccccccccccccccccccccccccccccccccccccccc" };

  // Open file descriptor with 755
  fd = open("./file.txt", O_RDWR|O_CREAT|O_TRUNC, 
    S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH);

  if (fd==-1) {
    perror("open: Failed");
    return 1;
  }

  // Complete iovec structures
  for (i=0; i<max; ++i) {
    iov[i].iov_base = buf[i];
    iov[i].iov_len = strlen(buf[i]);
  }

  // Write all by single call!
  n = writev(fd, iov, max);

  if (n==-1) {
    perror("writev: Failed");
    return 1;
  }

  printf("%d bytes written\n", n);

  // Close file descriptor
  if (close(fd)) {
    perror("close: Failed");
    return 1;
  }

  return 0;
}
