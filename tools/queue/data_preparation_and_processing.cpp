#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

std::mutex m;
std::queue<int> q;
std::vector<int> v;
std::condition_variable c;

void do_calc() {
  std::cout << __func__ << ": " << std::this_thread::get_id() << '\n';
  for (int i=0; i<10000; ++i)
    v.push_back(i);
}

void do_prep() {
  std::cout << __func__ << ": " << std::this_thread::get_id() << '\n';
  std::lock_guard<std::mutex> l(m);
  for (typename std::vector<int>::iterator it = v.begin(); it!=v.end(); ++it)
    q.push(*it);

  std::this_thread::sleep_for(std::chrono::seconds(3));
  c.notify_one();
}

void do_proc() {
  std::cout << __func__ << ": " << std::this_thread::get_id() << '\n';
  while (true) {
    std::unique_lock<std::mutex> l(m);
    c.wait(l, []{return !q.empty();});
    int data = q.front();
    q.pop();
    std::cout << data << '\n';
    if (q.empty())
      break;
  }
}

template <typename T>
void doit() {
  do_calc();
  do_prep();
  do_proc();
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
