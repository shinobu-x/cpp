#include "../hpp/high_res_clock.hpp"
#include "../hpp/shared_ptr.hpp"
#include "../hpp/tcp.hpp"

#include <boost/asio/read.hpp>
#include <boost/asio/ip/address.hpp>
#include <boost/asio/write.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <cstdio>
#include <cstdlib> /* size_t */
#include <cstring>
#include <vector>

#include <iostream>

const int num_samples = 100000;

struct transfer_all {
  typedef std::size_t result_type;
  result_type operator() (const boost::system::error_code& ec, result_type) {
    return (ec && ec != boost::asio::error::would_block) ? 0 : ~0;
  }
};

auto main() -> decltype(0)
{
  boost::system::error_code ec;
  boost::asio::ip::address addr;
  boost::asio::ip::address_v4 addr_value = addr.to_v4();
  unsigned short port = 12345;
  int max_connections = 3;
  std::size_t buf_size = 1024;
  bool spin = true;

  boost::asio::io_service ios;
  std::vector<boost::asio::detail::shared_ptr<tcp::socket> > sockets;

  for (int i = 0; i < max_connections; ++i) {
    try {
      boost::asio::detail::shared_ptr<tcp::socket> s(new tcp::socket(ios));
      tcp::endpoint target(addr_value, port);
      s->connect(target);
      s->set_option(tcp::no_delay(true));

      if (spin) {
        tcp::socket::non_blocking_io nbio(true);
        s->io_control(nbio);
      }

      sockets.push_back(s);
    } catch (boost::system::error_code& ec) {
      std::cout << ec << '\n';
    }
  }

  std::vector<unsigned char> write_buf(buf_size);
  std::vector<unsigned char> read_buf(buf_size);

  boost::posix_time::ptime start =
    boost::posix_time::microsec_clock::universal_time();
  boost::uint64_t start_hr = high_res_clock();

  boost::uint64_t samples[num_samples];


  for (int i = 0; i < num_samples; ++i) {
    tcp::socket& socket = *sockets[i % max_connections];

    boost::uint64_t t = high_res_clock();

    boost::system::error_code ec;
    boost::asio::write(socket,
      boost::asio::buffer(write_buf),
      transfer_all(), ec);

    boost::asio::read(socket,
      boost::asio::buffer(read_buf),
      transfer_all(), ec);

    samples[i] = high_res_clock() - t;

  }


  boost::posix_time::ptime stop =
    boost::posix_time::microsec_clock::universal_time();
  boost::uint64_t stop_hr = high_res_clock();
  boost::uint64_t elapsed_msec = (stop - start).total_microseconds();
  boost::uint64_t elapsed_hr = stop_hr - start_hr;
  double scale = 1.0 * elapsed_msec / elapsed_hr;
  int factor1 = 10 - 1;
  int factor2 = 100 - 1;
  int factor3 = 1000 - 1;

  std::sort(samples, samples + num_samples);
  std::printf("  0.0%%\t%f\n", samples[0]                           * scale); 
  std::printf("  0.1%%\t%f\n", samples[num_samples     / factor3]   * scale);
  std::printf("  1.0%%\t%f\n", samples[num_samples     / factor2]   * scale);
  std::printf(" 10.0%%\t%f\n", samples[num_samples     / factor1]   * scale);
  std::printf(" 20.0%%\t%f\n", samples[num_samples * 2 / factor1]   * scale);
  std::printf(" 30.0%%\t%f\n", samples[num_samples * 3 / factor1]   * scale);
  std::printf(" 40.0%%\t%f\n", samples[num_samples * 4 / factor1]   * scale);
  std::printf(" 50.0%%\t%f\n", samples[num_samples * 5 / factor1]   * scale);
  std::printf(" 60.0%%\t%f\n", samples[num_samples * 6 / factor1]   * scale);
  std::printf(" 70.0%%\t%f\n", samples[num_samples * 7 / factor1]   * scale);
  std::printf(" 80.0%%\t%f\n", samples[num_samples * 8 / factor1]   * scale);
  std::printf(" 90.0%%\t%f\n", samples[num_samples * 9 / factor1]   * scale);
  std::printf(" 99.0%%\t%f\n", samples[num_samples * 99 / factor2]  * scale);
  std::printf(" 99.9%%\t%f\n", samples[num_samples * 999 / factor3] * scale);
  std::printf("100.0%%\t%f\n", samples[num_samples - 1]             * scale);

  double total = 0.0;
  for (int i = 0;  i < num_samples; ++i)
    total += samples[i] * scale;
  std::printf("  mean\t%f\n", total / num_samples);

  return 0;
}  
