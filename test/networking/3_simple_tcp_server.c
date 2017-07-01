#include <stdio.h>
#include <unistd.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

int main()
{
  int sk_listen;
  int sk_write;
  struct sockaddr_in sadd_listen;
  struct sockaddr_in sadd_write;
  socklen_t len;

  sk_listen = socket(AF_INET, SOCK_STREAM, 0);

  sadd_listen.sin_family = AF_INET;
  sadd_listen.sin_port = htons(12345);
  sadd_listen.sin_addr.s_addr = INADDR_ANY;
//  inet_pton(AF_INET, "127.0.0.1", &sadd_listen.sin_addr.s_addr);
  bind(sk_listen, (struct sockaddr*)&sadd_listen, sizeof(sadd_listen));

  listen(sk_listen, 5);
  len = sizeof(sk_write);
  sk_write = accept(sk_listen, (struct sockaddr *)&sadd_write, &len);

  write(sk_write, "ABC", 3);

  close(sk_write);
  close(sk_listen);

  return 0;
}
