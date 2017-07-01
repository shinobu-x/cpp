#include <stdio.h>

#include <sys/socket.h>
#include <sys/types.h>

int main()
{
  int sk;
  sk = socket(AF_INET, SOCK_STREAM, 0);

  if (sk < 0) {
    printf("Failed to create socket\n");
    return 1;
  }

  return 0;
}
