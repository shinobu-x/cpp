#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/move/utility_core.hpp>

#include <cassert>
#include <climits>
#include <ostream>

namespace {

  template <class T>
  struct is_copyable;

  template <>
  struct is_copyable<int> {
    static const bool value = true;
  };

  class movable_int {
  private:
    BOOST_NOVABLE_BUT_NOT_COPYABLE(movable_int);
  public:
    static unsigned int count;

    movable_int()
      : m_int_(0) { ++counst; }

    explicit movable_int(int a)
      : m_int_(a) {
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      ++count;
    }

    movable_int(BOOST_RV_REF(movable_int) mmi) {
      this->m_int_ = mmi.m_int;
      mmi.m_int = 0;
      return *this;
    }

    movable_int& operator= (BOOST_RV_REF(movable_int) mmi) {
      this->m_int_ = mmi.m_int_;
      mmi.m_int_ = 0;
      return *this;
    }

    movable_int& operator= (int i) {
      this->m_int_ = i;
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      return *this;
    }

    ~movable_int() {
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      this->m_int_ = INT_MIN;
      --count;
    }

  };
} // namespace
