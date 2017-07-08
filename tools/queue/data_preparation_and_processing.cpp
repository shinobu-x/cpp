#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template <typename T>
class data_proc {
public:
  void do_calc() {
    calc_();
  }

  void do_prep() {
    prep_();
  }

  void do_proc() {
    proc_();
  }
private:
  std::mutex m_;
  std::queue<T> q_;
  std::vector<T> v_;
  std::condition_variable c_;

  void calc_() {
    for (T i=0; i<10000; ++i)
      v_.push_back(i);
  }

  void prep_() {
    std::lock_guard<std::mutex> l(m_);
    for (typename std::vector<T>::iterator it=v_.begin(); it!=v_.end(); ++it) \
      q_.push(*it);
//    std::this_thread::sleep_for(std::chrono::seconds(3));
    c_.notify_one();
  }

  void proc_() {
    while (true) {
      std::unique_lock<std::mutex> l(m_);
      c_.wait(l, [&]{ return !q_.empty(); });
      T data = q_.front();
      q_.pop();
      std::cout << data << '\n';
      if (q_.empty())
        break;
    }
  }
};

template <typename T>
void doit() {
  data_proc<T> go;
  go.do_calc();
  go.do_prep();
  go.do_proc();
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
