#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>

#include <arpa/inet.h>
#include <sys/socket.h>

int main()
{
  const char *name = "localhost";
  struct addrinfo addr_info_h, *addr_info_r;
  struct in_addr addr;
  char buf[32];
  int r;

  memset(&addr_info_h, 0, sizeof(addr_info_h));
  addr_info_h.ai_socktype = SOCK_STREAM;
  addr_info_h.ai_family = AF_INET;

  r = getaddrinfo(name, NULL, &addr_info_h, &addr_info_r);

  if (r != 0) {
    printf("Failed to get address(%d:%d)\n", r, errno);
    return -1;
  }

  addr.s_addr = ((struct sockaddr_in*)(addr_info_r->ai_addr))->sin_addr.s_addr;
  inet_ntop(addr_info_h.ai_family, &addr, buf, sizeof(buf));
  printf("IP Address: %s\n", buf);

  freeaddrinfo(addr_info_r);

  return 0;
}
