#include <errno.h>
#include <stdio.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

void show_port(int sk) {
  char buf[32];
  struct sockaddr_in saddr;
  socklen_t sk_len;
  sk_len = sizeof(saddr);

  // Get socket name
  if (getsockname(sk, (struct sockaddr*)&saddr, &sk_len) != 0) {
    perror("Error: Can't get socket name");
    return;
  }

  // IP to string
  inet_ntop(AF_INET, &saddr.sin_addr, buf, sizeof(buf));

  // Show result
  printf("%s:%d\n", buf, ntohs(saddr.sin_port));
}

int main()
{
  int sk_listen, sk_write;
  int r_l;
  ssize_t r_w;
  struct sockaddr_in saddr_write;
  socklen_t sk_len;

  // Create socket and listen
  sk_listen = socket(AF_INET, SOCK_STREAM, 0);
  r_l = listen(sk_listen, 5);

  if (r_l != 0) {
    printf("Failed to listen(%d:%d)\n", r_l, errno);
    return -1;
  }

  // Show port
  show_port(sk_listen);

  sk_len = sizeof(saddr_write);
  sk_write = accept(sk_listen, (struct sockaddr*)&saddr_write, &sk_len);

  if (sk_write < 0) {
    printf("Failed to create socket(%d:%d)\n", sk_write, errno);
    return -1;
  }

  r_w = write(sk_write, "Connect\n", 5);

  if (r_w != 0) {
    printf("Failed to write(%d:%d)\n", r_w, errno);
    return -1;
  }

  close(sk_write);
  close(sk_listen);

  return 0;
}
