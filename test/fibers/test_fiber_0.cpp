#include <boost/fiber/all.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

struct exception : public std::runtime_error {
  exception() : std::runtime_error("Oops") {}
};

struct dummy {
  dummy() = default;
  dummy(dummy cont&) = delete;
  dummy(dummy &&) = default;
  dummy& operator=(dummy const&) = delete;
  dummy& operator=(dummy&&) = default;
  int value;
};

auto main() -> decltype(0) {
  return 0;
}
