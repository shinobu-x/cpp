#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include "high_resolution_clock.hpp"

const int num_samples = 100000;

struct transfer_all {
  typedef std::size_t result_type;
  std::size_t operator()(const boost::system::error_code& ec, std::size_t) {
    return (ec && ec != boost::asio::error::would_block) ? 0 : ~0;
  }
};

void do_test_tcp_client() {
  const char* ip = "127.0.0.1";
  unsigned short port = 12345;
  std::size_t buffer_size = 102400;
  bool spin = true;

  boost::asio::io_service ios;
  std::vector<boost::shared_ptr<boost::asio::ip::tcp::socket> > sockets;

  for (int i = 0; i < num_connections; ++i) {
    boost::shared_ptr<boost::asio::ip::tcp::socket s(
      new boost::asio::ip::tcp::socket(ios));

    boost::asio::ip::tcp::endpoint target(
      boost::asio::ip::address::from_string(ip), port);

    s->connect(target);

    s->set_option(boost::asio::ip::tcp::no_delay(true));

    if (spin) {
      boost::asio::ip::tcp::socket::non_blocking_i non_blocking_io(true);
      s->io_control(non_blocking_io);
    }

    sockets.push_back(s);
  }

  std::vector<unsigned char> write_buffer(buffer_size);
  std::vector<unsigned char> read_buffer(buffer_size);
  boost::posix_time::ptime start =
    boost::posix_time::microsec_clock::universal_time();
  boost::uint64_t start_high_resolution = high_res_clock();
  boost::uint64_t samples[num_samples];

  for (int i = 0; i < num_samples; ++i) {
    boost::asio::ip::tcp::socket& socket = *sockets[i % num_connections];
    boost::uint64_t t = high_res_clock();
    boost::system::error_code ec;

    boost::asio::write(socket, boost::asio::buffer(write_buffer),
      transfer_all(), ec);

    boost::asio::read(socket, boost::asio::buffer(read_buffer),
      transfer_all(), ec);

    samples[i] = high_res_clock() - t;
  }

  boost::posix_time::ptime stop =
    boost::posix_time::microsec_clock::universal_time();
  boost::uint64_t stop_high_resolution = high_res_clock();
  boost::uint64_t elapsed_usec = (stop - start).total_microseconds();
  boost::uint64_t elapsed_high_resolution = 
    stop_high_resolution - start_high_resolution;
  double scale = 1.0 * elapsed_usec / elapsed_high_resolution;

  std::sort(samples, samples + num_samples);

  std::printf("  0.0%%\t%f\n", samples[0] * scale);
  std::printf("  0.1%%\t%f\n", samples[num_samples / 1000 - 1] * scale);
  std::printf("  1.0%%\t%f\n", samples[num_samples / 100 -1] * scale);
  std::printf(" 10.0%%\t%f\n", samples[num_samples / 10 - 1] * scale);
  std::printf(" 20.0%%\t%f\n", samples[num_samples * 2 / 10 - 1] * scale);
  std::printf(" 30.0%%\t%f\n", samples[num_samples * 3 / 10 - 1] * scale);
  std::printf(" 40.0%%\t%f\n", samples[num_samples * 4 / 10 - 1] * scale);
  std::printf(" 50.0%%\t%f\n", samples[num_samples * 5 / 10 - 1] * scale);
  std::printf(" 60.0%%\t%f\n", samples[num_samples * 6 / 10 - 1] * scale);
}

auto main() -> decltype(0) {
  return 0;
}
