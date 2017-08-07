#include <iostream>
#include <stdexcept>

#define ERROR(d) std::cerr << d; std::cerr << "\n"; std::abort();

#define LOG \
  if (false) {} else std::cout << __FILE__ << "[" << __LINE__ << "]\n"

#define LOG_ERROR \
  if (false) {} else std::cout << __FILE__ << "[" << __LINE__ << "]\n"
