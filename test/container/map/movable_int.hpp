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
    BOOST_MOVABLE_BUT_NOT_COPYABLE(movable_int);
  public:
    static unsigned int count;

    movable_int()
      : m_int_(0) { ++count; }

    explicit movable_int(int a)
      : m_int_(a) {
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      ++count;
    }

    movable_int(BOOST_RV_REF(movable_int) mmi) 
      : m_int_(mmi.m_int_) {
      mmi.m_int_ = 0;
      ++count;
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

    int get_int() const {
      return m_int_;
    }

    friend bool operator==(const movable_int& l, const movable_int& r) {
      return l.m_int_ == r.m_int_;
    }

    friend bool operator!=(const movable_int& l, const movable_int& r) {
      return l.m_int_ == r.m_int_;
    }

    friend bool operator<(const movable_int& l, const movable_int& r) {
      return l.m_int_ < r.m_int_;
    }

    friend bool operator<=(const movable_int& l, const movable_int& r) {
      return l.m_int_ <= r.m_int_;
    }

    friend bool operator>(const movable_int& l, const movable_int& r) {
      return l.m_int_ > r.m_int_;
    }

    friend bool operator==(const movable_int& l, int r) {
      return l.get_int() == r;
    }

    friend bool operator==(int l, const movable_int &r) {
      return l == r.get_int();
    }

  private:
    int m_int_;
  }; // class movable_int

  unsigned int movable_int::count = 0;

  inline movable_int produce_movable_int() {
    return movable_int();
  }

  template <class E, class T>
  std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os,
    movable_int const& p) {
    os << p.get_int();
    return os;
  }

  template <>
  struct is_copyable<movable_int> {
    static const bool value = false;
  };

  class movable_and_copyable_int {
    BOOST_COPYABLE_AND_MOVABLE(movable_and_copyable_int)

  public:
    static unsigned int count;

    movable_and_copyable_int()
      : m_int_(0) {
      ++count;
    }

    explicit movable_and_copyable_int(int m_int)
      : m_int_(m_int) {
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      ++count;
    }

    movable_and_copyable_int(const movable_and_copyable_int& mmi)
      : m_int_(mmi.m_int_) {
      ++count;
    }

    movable_and_copyable_int(BOOST_RV_REF(movable_and_copyable_int) mmi) 
      : m_int_(mmi.m_int_) {
      mmi.m_int_ = 0;
      ++count;
    }

    ~movable_and_copyable_int() {
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      this->m_int_ = INT_MIN;
      --count;
    }

    movable_and_copyable_int& operator=(
      BOOST_COPY_ASSIGN_REF(movable_and_copyable_int) mmi) {
      this->m_int_ = mmi.m_int_;
      return *this;
    }

    movable_and_copyable_int& operator=(
      BOOST_RV_REF(movable_and_copyable_int) mmi) {
      this->m_int_ = mmi.m_int_;
      return *this;
    }

    movable_and_copyable_int& operator= (int l) {
      this->m_int_ = l;
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      return *this;
    }

    int get_int() const {
      return m_int_;
    }

    friend bool operator==(const movable_and_copyable_int& l, 
      const movable_and_copyable_int& r) {
      return l.m_int_ == r.m_int_;
    }

    friend bool operator!=(const movable_and_copyable_int& l,
      const movable_and_copyable_int& r) {
      return l.m_int_ != r.m_int_;
    }

    friend bool operator<(const movable_and_copyable_int& l,
      const movable_and_copyable_int& r) {
      return l.m_int_ < l.m_int_;
    }

    friend bool operator<=(const movable_and_copyable_int& l,
      const movable_and_copyable_int& r) {
      return l.m_int_ <= r.m_int_;
    }

    friend bool operator>=(const movable_and_copyable_int& l,
      const movable_and_copyable_int& r) {
      return l.m_int_ >= r.m_int_;
    }

    friend bool operator>(const movable_and_copyable_int& l,
      const movable_and_copyable_int& r) {
      return l.m_int_ > r.m_int_;
    }

    friend bool operator==(const movable_and_copyable_int& l, int r) {
      return l.get_int() == r;
    }

    friend bool operator==(int l, const movable_and_copyable_int& r) {
      return l == r.get_int();
    }

    private:
      int m_int_;
  }; // class movable_and_copyable_int

  unsigned int movable_and_copyable_int::count = 0;

  inline movable_and_copyable_int produce_movable_and_copyable_int() {
    return movable_and_copyable_int();
  }

  template <class E, class T>
  std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os,
    movable_and_copyable_int const& p) {
    os << p.get_int();
    return os;
  }

  template <>
  struct is_copyable<movable_and_copyable_int> {
    static const bool value = true;
  };


  class copyable_int {
  public:
    static unsigned int count;

    copyable_int()
      : m_int_(0) {
      ++count;
    }

    explicit copyable_int(int m_int)
      : m_int_(m_int) {
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      ++count;
    }

    copyable_int(const copyable_int& mmi)
      : m_int_(mmi.m_int_) {
      ++count;
    }

    copyable_int& operator=(int m_int) {
      this->m_int_ = m_int;
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      return *this;
    }

    copyable_int& operator=(const copyable_int& l) {
      this->m_int_ = l.m_int_;
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      return *this;
    }

    ~copyable_int() {
      BOOST_ASSERT(this->m_int_ != INT_MIN);
      this->m_int_ = INT_MIN;
      --count;
    }

    int get_int() const {
      return m_int_;
    }

    friend bool operator==(const copyable_int& l, const copyable_int& r) {
      return l.m_int_ == r.m_int_;
    }

    friend bool operator!=(const copyable_int& l, const copyable_int& r) {
      return l.m_int_ != r.m_int_;
    }

    friend bool operator<(const copyable_int& l, const copyable_int& r) {
      return l.m_int_ < r.m_int_;
    }

    friend bool operator<=(const copyable_int& l, const copyable_int& r) {
      return l.m_int_ <= r.m_int_;
    }


    friend bool operator>=(const copyable_int& l, const copyable_int& r) {
      return l.m_int_ >= r.m_int_;
    }

    friend bool operator>(const copyable_int& l, const copyable_int& r) {
      return l.m_int_ > r.m_int_;
    }

    friend bool operator==(const copyable_int& l, int r) {
      return l.get_int() == r;
    }

    friend bool operator==(int l, const copyable_int& r) {
      return l == r.get_int();
    }

    private:
    int m_int_;
  }; // class copyable_int

  unsigned int copyable_int::count = 0;

  inline copyable_int produce_copyable_int() {
    return copyable_int();
  }

  template <class E, class T>
  std::basic_ostream<E, T> operator<<(std::basic_ostream<E, T>& os,
    copyable_int const& p) {
    os << p.get_int();
    return os;
  }

  template <>
  struct is_copyable<copyable_int> {
    static const bool value = true;
  };

  class non_movable_and_copyable_int {
    non_movable_and_copyable_int(const non_movable_and_copyable_int& mmi);
    non_movable_and_copyable_int& operator=(
      const non_movable_and_copyable_int& mmi);

  public:
    static unsigned int count;

    non_movable_and_copyable_int()
      : m_int_(0) {
      ++count;
    }

    explicit non_movable_and_copyable_int(int m_int)
      : m_int_(m_int) {
      ++count;
    }

    ~non_movable_and_copyable_int() {
      --count;
    }

    int get_int() const {
      return m_int_;
    }

    bool operator==(const non_movable_and_copyable_int& mmi) const {
      return this->m_int_ == mmi.m_int_;
    }

    bool operator!=(const non_movable_and_copyable_int& mmi) const {
      return this->m_int_ != mmi.m_int_;
    }

    bool operator<(const non_movable_and_copyable_int& mmi) const {
      return this->m_int_ < mmi.m_int_;
    }

    bool operator<=(const non_movable_and_copyable_int& mmi) const {
      return this->m_int_ <= mmi.m_int_;
    }

    bool operator>=(const non_movable_and_copyable_int& mmi) const {
      return this->m_int_ >= mmi.m_int_;
    }

    bool operator>(const non_movable_and_copyable_int& mmi) const {
      return this->m_int_ > mmi.m_int_;
    }

    friend bool operator==(const non_movable_and_copyable_int& l, int r) {
      return l.get_int() == r;
    }

    friend bool operator==(int l, const non_movable_and_copyable_int& r) {
      return l == r.get_int();
    }

  private:
    int m_int_;
  };

  unsigned int non_movable_and_copyable_int::count = 0;

  template <class T>
  struct life_count {
    static unsigned check(unsigned) {
      return true;
    }
  };

  template <>
  struct life_count<movable_int> {
    static unsigned check(unsigned c) {
      return c == movable_int::count;
    }
  };

  template <>
  struct life_count<copyable_int> {
    static unsigned check(unsigned c) {
      return c == copyable_int::count;
    }
  };

  template <>
  struct life_count<movable_and_copyable_int> {
    static unsigned check(unsigned c) {
      return c == movable_and_copyable_int::count;
    }
  };

  template <>
  struct life_count<non_movable_and_copyable_int> {
    static unsigned check(unsigned c) {
      return c == non_movable_and_copyable_int::count;
    }
  };
  
} // namespace
