#include <cstdlib>
#include <iostream>
#include <unistd.h>    /** fork **/

#include <sys/types.h> /** waitpid **/
#include <sys/wait.h>  /** waitpid **/

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

class server {
public:
  server(boost::asio::io_service& ios, unsigned short port)
    : ios_(ios),
      sig_(ios, SIGCHLD),
      ap_(ios,
        boost::asio::ip::tcp::endpoint(
          boost::asio::ip::tcp::v4(), port)),
      sk_(ios) {}

  void do_wait() {
    wait_();
  }
  void do_accept() {
    accept_();
  }
 

private:
  void wait_() {
    // Start an asynchronous wait for one of the signals to occur
    sig_.async_wait(boost::bind(&server::handle_wait_, this));
  }

  void handle_wait_() {
    // Determine whether the acceptor is open
    if (ap_.is_open()) {
      int status = 0;
      // Wait for process to change state
      // WNOHANG:
      // Return immediately if no child has exited
      while (waitpid(-1, &status, WNOHANG) > 0) {}

      wait_();
    }
  }

  void accept_() {
    ap_.async_accept(sk_,
      boost::bind(&server::handle_accept_, this, _1));
  }

  void handle_accept_(const boost::system::error_code& ec) {
    if(!ec) {
      // Notify the io_service that the process is about to fork
      ios_.notify_fork(boost::asio::io_service::fork_prepare);

      if (fork() == 0) {
        // Notify the io_service that the process has forked and is the child
        ios_.notify_fork(boost::asio::io_service::fork_child);

        ap_.close();
        sig_.cancel();

        read_();
      } else {
        // Notify the io_service that the process has forked and is the parent
        ios_.notify_fork(boost::asio::io_service::fork_parent);

        ap_.close();
        accept_();
      }
    } else
      std::cerr << "Error: " << ec.message() << '\n';
  }

  void read_() {
    sk_.async_read_some(boost::asio::buffer(data_),
      boost::bind(&server::handle_read_, this, _1, _2));
  }

  void handle_read_(const boost::system::error_code& ec, std::size_t len) {
    boost::asio::async_write(
      sk_,
      boost::asio::buffer(data_, len),
      boost::bind(&server::handle_write_, this, _1));
  }

  void handle_write_(const boost::system::error_code& ec) {
    if (!ec)
      read_();
  }

// Member variables
  boost::asio::io_service& ios_;
  // Construct a signal set registered for process termination
  boost::asio::signal_set sig_;
  boost::asio::ip::tcp::acceptor ap_;
  boost::asio::ip::tcp::socket sk_;
  boost::array<char, 1024> data_;
};

auto main() -> decltype(0)
{
}
