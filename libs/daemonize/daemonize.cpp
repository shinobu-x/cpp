#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>

#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <sysexits.h>
#include <unistd.h>
#include <syslog.h>

#define MAX_FD 64

int daemonize(int chdir_flag) {

  // Status:
  // There is a parent process, waiting for TTY

  // Makes child process and detach it from parent process
  pid_t pid = fork();
  if (pid == -1) {
    return -1;
  // Exits from parent process
  } else if (pid != 0) {
   // Calls _exit() to prevent user defined cleanup method from being called
    _exit(0);
  }

  // Status:
  // There is a child process keeping TTY

  // Detaches TTY, make process session / process leader
  setsid();

  // Ignore HUP signal to prevent child process from being killed when parent
  // process is dead
  signal(SIGHUP, SIG_IGN);

  // Status:
  // A chiled process is still session leader. If TTY is open, this TTY is
  // linked to this child process

  // Detaches parent process
  pid = fork();
  if (pid == 0) {
    // Stop child process
    _exit(0);
  }

  // Prepare deamonize
  if (chdir_flag == 0) {
    chdir("/");
  }

  // Closes all file descriptor derived from parent process
  for (int i = 0; i < MAX_FD; ++i) {
    close(i);
  }

  // Opens  stdin, stdout, stderr with /dev/null
  int fd = open("/dev/null", O_RDWR, 0);
  if (fd != -1) {
    // Copy file descriptor
    // Makes 0, 1, 2 hold by child process point to /dev/null
    dup2(fd, 0);
    dup2(fd, 1);
    dup2(fd, 2);

    if (fd < 2) {
      close(fd);
    }
  }

  return 0;
}

auto main() -> decltype(0) {
  char buf[256];
  daemonize(1);

  // Display current directory
  syslog(LOG_USER | LOG_NOTICE, "daemon:cwd=%s\n", getcwd(buf, sizeof(buf)));

  return 0;
}
