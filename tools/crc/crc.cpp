#include <cstdio>
#include <cstring>
#include <iostream>

#include <arpa/inet.h>

#include "crc.hpp"

#define M_MAXN 255

template <typename T>
struct value_type;

template <>
struct value_type<uint16_t> {
  const static int value = 1;
};

template <>
struct value_type<uint32_t> {
  const static int value = 2;
};

template <typename T, int L>
struct crc_check_t {
private:
  int l_;
  unsigned long i_ = time(NULL);
  typedef uint8_t byte_;
  byte_ bf_[32];
  typedef T type_;
  type_ crc_;

public:
  crc_check_t() : l_(L) {}

  int type = value_type<T>::value;

  void cal_crc() {
    if (type == 1) {
      typedef uint16_t type_;
      type_ crc_ = 0xFFFF;
    } else {
      typedef uint32_t type_;
      type_ crc_ = 0xFFFFFFFF;
    }
    T crc_type;
    do_gen_rand();
    crc_type = do_cal();
  }

  int gen_rand(int max, unsigned long* i) {
    *i = (*i)*1103515245 + 12345;
    double n = max*((double)(*i/65536 % 32768) / (double)(32767));
    return (n > 0) ? ((int)(n + 0.5)) : ((int)(n - 0.5));
  }

  void do_gen_rand() {
    for (int i = 0; i < l_; ++i)
      bf_[i] = gen_rand(M_MAXN, &i_);
    show_rand();
  }

  void show_rand() {
    for (int i = 0; i < l_; ++i)
      printf("%02x", bf_[i]);
    std::cout << '\n';
  }

private:
  T do_cal() {
    if (type == 1) {  /// uint16_t
      for (int i = 0; i < l_; ++i) {
        crc_ ^= (type_)(bf_[i] << 8);
        for (int j = 0; j < 8; ++j) {
          if (crc_ & 0x8000) {
            crc_ <<= 1;
            crc_ ^= 0x1021;
          } else
            crc_ <<= 1;
        }
      }
      crc_ = htons(crc_);

      memcpy(bf_+28, &crc_, sizeof(crc_));
    }
    return crc_;
  }
};

template <typename T>
void doit() {
  crc_check_t<T, 28> crc_type;

  crc_type.cal_crc();
}
auto main() -> int
{
  doit<uint16_t>();
  return 0;
}
