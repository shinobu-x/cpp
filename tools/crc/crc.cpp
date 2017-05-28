#include <iostream>

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
  typedef uint8_t byte;
  int type = value_type<T>::value;

  void cal_crc() {
    if (type == 1) {
      typedef uint16_t type_;
      type_ crc_ = 0xFFFF;
    } else {
      typedef uint32_t type_;
      type_ crc_ = 0xFFFFFFFF;
    }
    do_cal();
  }

  void do_cal() {
    std::cout << "T" << '\n';
  }
private:
  typedef T type_;
  type_ crc_;
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
