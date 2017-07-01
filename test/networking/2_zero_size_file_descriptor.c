#include <stdio.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/types.h>

int main()
{
  int sk;
  printf("fileno(stdin) = %d\n", fileno(stdin));

  close(0);

  sk = socket(AF_INET, SOCK_DGRAM, 0);
  printf("socket=%d\n", sk);

  return 0;
}
