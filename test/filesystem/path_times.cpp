#include <boost/timer/timer.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/config.hpp>

#include <iostream>

namespace {
  boost::int64_t max_cycles;

  template <class string_type>
  boost::timer::nanosecond_type time_ctor(const string_type& s) {
    boost::timer::auto_cpu_timer t;
    boost::int64_t c = 0;
    do {
      boost::filesystem::path p(s);
      ++c;
    } while (c < max_cycles);

    boost::timer::cpu_times e = t.elapsed();
    return e.user + e.system;
  }

  boost::timer::nanosecond_type time_loop() {
    boost::timer::auto_cpu_timer t;
    boost::int64_t c = 0;
    do {
      ++c;
    } while (c < max_cycles);

    boost::timer::cpu_times e = t.elapsed();
    return e.user + e.system;
  }
} // namaspace

auto main(int argc, char** argv) -> decltype(0) {
  if (argc != 2) {
    std::cout << "Usage: <program name> <cycle-in-millions>\n";
    return 1;
  }

  max_cycles = std::atoi(*(argv+1)) * 1000000LL;
  std::cout << "Testing " << std::atoi(*(argv+1)) << " million cycles" << '\n';

  std::cout << "time_loop\n";
  boost::timer::nanosecond_type x = time_loop();

  std::cout << "time_ctor with string\n";
  boost::timer::nanosecond_type s = time_ctor(std::string("/foo/bar/baz"));

  std::cout << "time_ctor with wstring\n";
  boost::timer::nanosecond_type w = time_ctor(std::wstring(L"/foo/bar/baz"));

  if (s > w)
    std::cout << "Narrow/Wide CPU Time Ratio = " << (long double)(s)/w << '\n';
  else
    std::cout << "Wide/Narrow CPU Time Ratio = " << (long double)(w)/s << '\n';

  return 0;
}
