#include <boost/system/error_code.hpp>
#include <boost/cerrno.hpp>
#include <string>
#include <cstdio>

#ifdef BOOST_POSIX_API
# include <sys/stat.h>
#else
# include <windows.h>
#endif

boost::system::error_code do_mkdir(const std::string& path) {
  return boost::system::error_code(
#ifdef BOOST_POSIX_API
  ::mkdir(path.c_str(), S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH) == 0 ? 0 : errno,
#else
  ::CreateDirectoryA(path.c_str(), 0) != 0 ? 0 : ::GetLastError(),
#endif
  boost::system::system_category()
  );
}

boost::system::error_code do_remove(const std::string& path) {
  return boost::system::error_code(
    std::remove(path.c_str()) == 0 ? 0 : errno,
    boost::system::generic_category()
  );
}

#include <iostream>

auto main() -> decltype(0) {
  boost::system::error_code ec = do_mkdir("/no/such/file/or/directory");
  std::cout << ec.value() << '\n';

  ec = do_remove("/no/such/file/or/directory");
  std::cout << ec.value() << '\n';
}
