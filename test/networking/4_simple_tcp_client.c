#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

int main()
{
  struct sockaddr_in sadd_server;
  int sk_read;
  char buf[32];
  int n;

  sk_read = socket(AF_INET, SOCK_STREAM, 0);

  sadd_server.sin_family = AF_INET;
  sadd_server.sin_port = htons(12345);
  inet_pton(AF_INET, "127.0.0.1", &sadd_server.sin_addr.s_addr);
  connect(sk_read, (struct sockaddr*)&sadd_server, sizeof(sadd_server));

  memset(buf, 0, sizeof(buf));
  n = read(sk_read, buf, sizeof(buf));

  printf("%d, %s\n", n, buf);

  close(sk_read);
  return 0;
}
