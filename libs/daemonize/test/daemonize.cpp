#include <iostream>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <syslog.h>

void daemonize() {

  if(getppid() == 1) {
    return;
  }

  int r = fork();

  if (r < 0) {
    exit(1);
  }
  if (r > 0) {
    exit(0);
  }

  setsid();

  int fd;
  for (fd = getdtablesize(); fd >= 0; --fd) {
    close(fd);
  }

  fd = open("/dev/null",O_RDWR);
  dup(fd);
  dup(fd);
  umask(027);

  const char* path = "/tmp";
  const char* lock_file = getpid() + ".lock";
  int lock;
  chdir(path);
  lock = open(lock_file, O_RDWR|O_CREAT, 0640);

  if (lock < 0) {
    exit(1);
  }

  if (lockf(lock, F_TLOCK, 0) < 0) {
    exit(0);
  }
std::cout << __LINE__ << '\n';
  char pid[10];
  sprintf(pid,"%d\n",getpid());
  write(lock, pid, strlen(pid));

  signal(SIGCHLD,SIG_IGN);
  signal(SIGTSTP,SIG_IGN);
  signal(SIGTTOU,SIG_IGN);
  signal(SIGTTIN,SIG_IGN);
  signal(SIGHUP,SIG_IGN);
  signal(SIGTERM,SIG_IGN);
}


#define LOCK_FILE   "exampled.lock"
void daemonize(
  const char* name,
  const char* dir,
  const char* in,
  const char* out,
  const char* err) {

  if (!name) {
    name = "exampled";
  }
  if (!dir) {
    dir = "/tmp";
  }
  if (!in) {
    in = "/dev/null";
  }
  if (!out) {
    out = "/dev/null";
  }
  if (!err) {
    err = "/dev/null";
  }

  int fd,lock;
  char pid[10];

  // Already a daemon
  if(getppid() == 1) {
    return;
  }

  int child = fork();
  if (child < 0) {
    exit(1); // Fork error
  }
  if (child > 0) {
    exit(0); // Parent exits
  }
  // Child continues
  setsid(); // Obtains a new process group
  for (fd = getdtablesize(); fd >= 0; --fd) {
    close(fd); // Closes all descriptors
  }

  stdin = fopen(in, "r");
  stdout = fopen(out, "w+");
  stderr = fopen(err, "w+");

  // Handles standard I/O
  fd = open("/dev/null", O_RDWR);
  dup(fd);
  dup(fd);

  // Sets newly created file permissions
  umask(027);
  // Changes running directory
  chdir("/tmp");
  lock = open(LOCK_FILE ,O_RDWR|O_CREAT, 0640);
  if (lock < 0) {
    exit(1); // Can not open
  }
  if (lockf(lock, F_TLOCK, 0) < 0) {
    exit(0); // Can not lock
  }
  // first instance continues 
  sprintf(pid, "%d\n", getpid());
  write(lock, pid, strlen(pid)); // record pid to lockfile
  signal(SIGCHLD, SIG_IGN); // ignore child
  signal(SIGTSTP, SIG_IGN); // ignore tty signals
  signal(SIGTTOU, SIG_IGN);
  signal(SIGTTIN, SIG_IGN);
  signal(SIGHUP, SIG_IGN); // catch hangup signal
  signal(SIGTERM, SIG_IGN); // catch kill signal
}

static int daemonize(const char* lockfile)
{
  pid_t pid, sid, parent;
  int lfp = -1;
  char buf[16];

  // already a daemon
  if (getppid() == 1) {
    return 1;
  }

  // Each copy of the daemon will try to create a file and write its process ID
  // in it. This will allow administrators to identify the process easily
  // Create the lock file as the current user
  if (lockfile && lockfile[0]) {
    lfp = open(lockfile, O_RDWR|O_CREAT, 0640);
    if (lfp < 0) {
      syslog(LOG_ERR, "unable to create lock file %s, code=%d (%s)",
        lockfile, errno, strerror(errno));
      exit(EXIT_FAILURE);
    }
  }

  // If the file is already locked, then to ensure that
  // only one copy of record is running. The filelock function will fail
  // with errno set to EACCESS or EAGAIN.
  //if (filelock(lfp) < 0) {
  //  if (errno == EACCES || errno == EAGAIN) {
  //    close(lfp);
  //    exit(EXIT_FAILURE);
  //  }
  //  syslog(LOG_ERR, "can't lock %s: %s", lockfile, strerror(errno));
  //  exit(EXIT_FAILURE);
  //}
  ftruncate(lfp, 0);
  sprintf(buf, "%ld", (long)getpid());
  write(lfp, buf, strlen(buf)+1);

  // Trap signals that we expect to recieve
  signal(SIGCHLD, SIG_IGN);
  signal(SIGUSR1, SIG_IGN);
  signal(SIGALRM, SIG_IGN);

  // Fork off the parent process
  pid = fork();
  if (pid < 0) {
    syslog(LOG_ERR, "unable to fork daemon, code=%d (%s)",
      errno, strerror(errno));
    exit(EXIT_FAILURE);
  }
  // If we got a good PID, then we can exit the parent process.
  if (pid > 0) {
    // Wait for confirmation from the child via SIGTERM or SIGCHLD, or
    // for two seconds to elapse (SIGALRM).  pause() should not return.
    alarm(2);
    pause();

    exit(EXIT_FAILURE);
  }

  // At this point we are executing as the child process
  parent = getppid();

  // Cancel certain signals
  signal(SIGCHLD,SIG_DFL); // A child process dies
  signal(SIGTSTP,SIG_IGN); // Various TTY signals
  signal(SIGTTOU,SIG_IGN);
  signal(SIGTTIN,SIG_IGN);
  signal(SIGHUP, SIG_IGN); // Ignore hangup signal
  signal(SIGTERM,SIG_DFL); // Die on SIGTERM

  // Change the file mode mask
  umask(0);

  // Create a new SID for the child process
  sid = setsid();
  if (sid < 0) {
    syslog(LOG_ERR, "unable to create a new session, code %d (%s)",
      errno, strerror(errno));
    exit(EXIT_FAILURE);
  }

  // Change the current working directory.  This prevents the current
  // directory from being locked; hence not being able to remove it.
  if ((chdir("/tmp")) < 0) {
    syslog( LOG_ERR, "unable to change directory to %s, code %d (%s)",
      "/", errno, strerror(errno) );
    exit(EXIT_FAILURE);
  }

  // Redirect standard files to /dev/null
  freopen("/dev/null", "r", stdin);
  freopen("/dev/null", "w", stdout);
  freopen("/dev/null", "w", stderr);

  // Tell the parent process that we are A-okay
  kill(parent, SIGUSR1);
  return 0;
}
auto main() -> decltype(0) {

  const char* lock_file = "process.lock";
  daemonize(lock_file);

  while (1) {
    sleep(100000000);
/*
    std::cout << "...\n";
    syslog(LOG_NOTICE, "Daemon ttl %d", ttl);
    sleep(delay);
    ttl -= delay;
*/
  }

  return EXIT_SUCCESS;
}
