#include <boost/move/core.hpp>
#include <boost/move/traits.hpp>

class movable {
  BOOST_MOVABLE_BUT_NOT_COPYABLE(movable);
  int value_;

public:
  movable() : value_(1) {}

  movable(BOOST_RV_REF(movable) m) {
    value_ = m.value_;
    m.value_ = 0;
  }

  movable& operator=(BOOST_RV_REF(movable) m) {
    value_ = m.value_;
    m.value_ = 0;
    return *this;
  }

  bool moved() const {
    return !value_;
  }

  int value() const {
    return value_;
  }
};

namespace boost {
template <>
struct has_nothrow_move<movable> {
  static const bool value = true;
};
} // namespace
