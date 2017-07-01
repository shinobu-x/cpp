#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <sys/socket.h>

int main()
{
  int sk;
  ssize_t r;

  sk = socket(AF_INET, 1234, 5678);

  r = write(sk, "ABC", 3);

  if (r != 0) {
    printf("Failed to write(%d:%d)\n", r, errno);
    return -1;
  }

  return 0;
}
