#include <boost/throw_exception.hpp>

#include "../macro/config.hpp"

class my_exception
  : public std::exception {
public:
  my_exception() {
    std::cout << "Exception" << '\n';
  }
};

auto main() -> decltype(0) {
  try {
    boost::throw_exception(my_exception());
    ERROR("boost::throw_exception failed to throw.");
  } catch (my_exception&) {}
}
