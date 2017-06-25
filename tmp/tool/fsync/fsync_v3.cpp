#include <sys/stat.h>
#include <sys/types.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <thread>
#include <iostream>
#include <vector>

#define M_BUFFER_SIZE 1024

template <typename T>
struct fsync_t {

  void do_set() {
    if ((str_ = (char*)(malloc(M_BUFFER_SIZE))) == NULL)
      perror("Error: malloc");

    if ((fd_ = creat(file_, S_IWUSR)) < 0)
      perror("Error: creat");

    memset(str_, '*', M_BUFFER_SIZE);
  }

  void do_fsync() {
    int r;
    if ((r = write(fd_, str_, M_BUFFER_SIZE)) == -1)
      perror("Error: write");
    else
      w_cnt_++;
      
    if (fsync(fd_) != 0)
      perror("Error: fsync");
    else
      f_cnt_++;

    this->do_out();
  }

  void do_out() {
    std::cout << "Write: " << w_cnt_ << '\n';
    std::cout << "Fsync: " << f_cnt_ << '\n';
  }

  void do_thread() {
    T th([]{std::cout << "Do some stuff..." << '\n';});
    th.join();
  }

private:
  int fd_;
  char *str_;
  int w_cnt_ = 0, f_cnt_ = 0;
  const char file_[14] = "/tmp/test.txt";
};

template <typename T>
T doit() {
  fsync_t<std::thread> ft;
  std::vector<fsync_t<std::thread> > fv;

  ft.do_set();
  ft.do_fsync();
  ft.do_thread();
}

auto main() -> int
{
  doit<int>();
  return 0;
}
