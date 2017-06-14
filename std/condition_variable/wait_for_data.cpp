#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

struct process_data {
private:
  std::mutex m_;
  std::condition_variable cv_;
  std::vector<int> v_;
  bool data_ready_ = false;

  void ready() {
    for (std::vector<int>::iterator it = v_.begin(); it != v_.end(); ++it)
      std::cout << *it << '\n';
  }
public:
  void preparing() {
    {
      std::lock_guard<std::mutex> lg(m_);

      for (int i = 0; i < 100; ++i)
        v_.push_back(i);

      std::this_thread::sleep_for(std::chrono::seconds(1));
      data_ready_ = true;
    }
    cv_.notify_one();
  }

  void prepared() {
    std::unique_lock<std::mutex> ul(m_);

    cv_.wait(ul, [this] { return data_ready_; });

    ready();
  }
};

auto main() -> int
{
  process_data pd;

  std::thread t1([&pd]{ pd.preparing(); });
  std::thread t2([&pd]{ pd.prepared(); });

  t2.join();
  t1.join();

  return 0;
}
