#include <iostream>
#include <thread>
#include <chrono>

template <typename T = void>
T job1() {
  std::this_thread::sleep_for(std::chrono::seconds(2));
}

template <typename T = void>
T job2() {
  std::this_thread::sleep_for(std::chrono::seconds(3));
}

template <typename T>
T doit() {
  std::thread t1(job1<>);
  std::thread t2(job2<>);

  std::cout << "Thread1 ID: " << t1.get_id() << '\n';
  std::cout << "Thread2 ID: " << t2.get_id() << '\n';

  std::swap(t1, t2);

  std::cout << "Thread1 ID: " << t1.get_id() << '\n';
  std::cout << "Thread2 ID: " << t2.get_id() << '\n';

  t1.swap(t2);

  std::cout << "Thread1 ID: " << t1.get_id() << '\n';
  std::cout << "Thread2 ID: " << t2.get_id() << '\n';

  t1.join();
  t2.join();

  return 0;
}

auto main() -> int
{
  doit<int>();
  return 0;
}
