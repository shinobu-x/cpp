#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/system/error_code.hpp>
#include <boost/thread.hpp>

#include <iostream>
#include <mutex>

class thread_pool {
private:
  boost::asio::io_service ios_;
  boost::asio::io_service::work work_;
  boost::thread_group thread_group_;
  std::size_t available_;
  boost::mutex m_;

public:
  thread_pool(std::size_t pool_size)
    : work_(ios_),
      available_(pool_size) {
    for (std::size_t i = 0; i < pool_size; ++i) 
      thread_group_.create_thread(
        boost::bind(&boost::asio::io_service::run, &ios_)); 
  }

  ~thread_pool() {
    ios_.stop();

    try {
      thread_group_.join_all();
    } catch (boost::system::error_code ec) {
    }
  }

  template <typename Function>
  void do_work(Function work) {
    boost::unique_lock<boost::mutex> l(m_);
    if (0 == available_)
      return;
    --available_;
    ios_.post(boost::bind(&thread_pool::hander_, this,
      boost::function<void()>(work)));
  }
private:
  void hander_(boost::function<void()> func) {
    try {
      func();
    } catch (boost::system::error_code ec) {
    } 
    {
      boost::unique_lock<boost::mutex> l(m_);
      ++available_;
    }
  }
};

void work1() {
  std::cout << "1" << '\n';
}

struct work2 {
  void operator() (){
    std::cout << "2" << '\n';
  } 
};

void work3(int) { 
  std::cout << "3" << '\n';
}

auto main() -> decltype(0) {
  thread_pool tp(2);

  tp.do_work(work1);
  tp.do_work(work2());
  tp.do_work(boost::bind(&work3, 3));

  return 0;
}
