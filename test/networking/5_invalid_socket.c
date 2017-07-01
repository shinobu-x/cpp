#include <errno.h>
#include <stdio.h>

#include <sys/socket.h>

int main()
{
  int sk;
  sk = socket(1234, 5678, 9);

  if (sk < 0) {
    printf("Failed to create socket(%d)\n", errno);
    return -1;
  }

  return 0;
}
