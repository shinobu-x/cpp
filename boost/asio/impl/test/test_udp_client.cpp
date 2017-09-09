#include <boost/asio/ip/udp.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "high_resolution_clock.hpp"

const int num_samples = 1000000;

void do_test_udp_client() {

  unsigned short first_port = 12345;
  unsigned short num_ports = 1000;
  std::size_t buffer_size = 1024;
  bool spin = true;

  boost::asio::io_service ios;
  boost::asio::ip::udp::socket s(ios, boost::asio::ip::udp::endpoint(
    boost::asio::ip::udp::v4(), 0));

  if (spin) {
    boost::asio::ip::udp::socket::non_blocking_io non_blocking_io(true);
    s.io_control(non_blocking_io);
  }

  boost::asio::ip::udp::endpoint target(
    boost::asio::ip::address_v4::loopback(), first_port);
  unsigned short last_port = first_port + num_ports - 1;
  std::vector<unsigned char> write_buffer(buffer_size);
  std::vector<unsigned char> read_buffer(buffer_size);

  boost::posix_time::ptime start =
    boost::posix_time::microsec_clock::universal_time();
  boost::uint64_t start_high_resolution = high_res_clock();
  boost::uint64_t samples[num_samples];

  for (int i = 0; i < num_samples; ++i) {
    boost::uint64_t t = high_res_clock();
    boost::system::error_code ec;
    s.send_to(boost::asio::buffer(write_buffer), target, 0, ec);
    do
      s.receive(boost::asio::buffer(read_buffer),  0, ec);
    while (ec == boost::asio::error::would_block);

    samples[i] = high_res_clock() - t;

    if (target.port() == last_port)
      target.port(first_port);
    else
      target.port(target.port() + 1);
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
  std::printf("  1.0%%\t%f\n", samples[num_samples / 100 - 1] * scale);
  std::printf(" 10.0%%\t%f\n", samples[num_samples / 10 - 1] * scale);
  std::printf(" 20.0%%\t%f\n", samples[num_samples * 2 / 10 - 1] * scale);
  std::printf(" 30.0%%\t%f\n", samples[num_samples * 3 / 10 - 1] * scale);
  std::printf(" 40.0%%\t%f\n", samples[num_samples * 4 / 10 - 1] * scale);
  std::printf(" 50.0%%\t%f\n", samples[num_samples * 5 / 10 - 1] * scale);
  std::printf(" 60.0%%\t%f\n", samples[num_samples * 6 / 10 - 1] * scale);
  std::printf(" 70.0%%\t%f\n", samples[num_samples * 7 / 10 - 1] * scale);
  std::printf(" 80.0%%\t%f\n", samples[num_samples * 8 / 10 - 1] * scale);
  std::printf(" 90.0%%\t%f\n", samples[num_samples * 9 / 10 -1 ] * scale);
  std::printf(" 99.0%%\t%f\n", samples[num_samples * 99 / 100 - 1] * scale);
  std::printf(" 99.9%%\t%f\n", samples[num_samples * 999 / 1000 - 1] * scale);
  std::printf("100.0%%\t%f\n", samples[num_samples - 1] * scale);

  double total = 0.0;
  for (int i = 0; i < num_samples; ++i)
    total += samples[i] * scale;
  std::printf("  mean\t%f\n", total / num_samples);
}

auto main() -> decltype(0) {
  do_test_udp_client();
  return 0;
}
