#include <boost/asio/ip/address_v4.hpp>

#include <sstream>

void test_1() {
  try {
    boost::system::error_code ec;

    boost::asio::ip::address_v4 addr1;
    const boost::asio::ip::address_v4::bytes_type const_bytes_value =
      {{127, 0, 0, 1}};
    boost::asio::ip::address_v4 addr2(const_bytes_value);
    const unsigned long const_ulong_value = 0x7F000001;
    boost::asio::ip::address_v4 addr3(const_ulong_value);

    assert(!addr1.is_loopback());
    assert(addr1.is_unspecified());
    assert(addr1.is_class_a());
    assert(!addr1.is_class_b());
    assert(!addr1.is_class_c());
    assert(!addr1.is_multicast());

    boost::asio::ip::address_v4::bytes_type bytes_type_value =
      addr1.to_bytes();
    (void)bytes_type_value;

    unsigned long ulong_value = addr1.to_ulong();
    (void)ulong_value;

    std::string string_value = addr1.to_string();
    string_value = addr1.to_string(ec);

    addr1 = boost::asio::ip::address_v4::from_string("127.0.0.1");
    addr1 = boost::asio::ip::address_v4::from_string("127.0.0.1", ec);
    addr1 = boost::asio::ip::address_v4::from_string(string_value);
    addr1 = boost::asio::ip::address_v4::from_string(string_value, ec);

    addr1 = boost::asio::ip::address_v4::any();
    addr1 = boost::asio::ip::address_v4::loopback();
    addr1 = boost::asio::ip::address_v4::broadcast();
    addr1 = boost::asio::ip::address_v4::broadcast(addr2, addr3);
    addr1 = boost::asio::ip::address_v4::netmask(addr2);

    assert(!(addr1 == addr2));
    assert(addr1 != addr2);
    assert(!(addr1 < addr2));
    assert(addr1 > addr2);
    assert(!(addr1 <= addr2));
    assert(addr1 >= addr2);

    std::ostringstream os;
    os << addr1;

#if !defined(BOOST_NO_STD_WSTREAMBUF)
    std::wostringstream wos;
    wos << addr1;
#endif
  } catch (std::exception) {
  }
}

void test_2() {
  boost::asio::ip::address_v4 a1;

  assert(a1.to_bytes()[0] == 0);
  assert(a1.to_bytes()[1] == 0);
  assert(a1.to_bytes()[2] == 0);
  assert(a1.to_bytes()[3] == 0);
  assert(a1.to_ulong() == 0);

  boost::asio::ip::address_v4::bytes_type b1 = {{1, 2, 3, 4}};
  boost::asio::ip::address_v4 a2(b1);

  assert(a2.to_bytes()[0] == 1);
  assert(a2.to_bytes()[1] == 2);
  assert(a2.to_bytes()[2] == 3);
  assert(a2.to_bytes()[3] == 4);
  assert(((a2.to_ulong() >> 24) & 0xFF) == b1[0]);
  assert(((a2.to_ulong() >> 16) & 0xFF) == b1[1]);
  assert(((a2.to_ulong() >> 8) & 0xFF) == b1[2]);
  assert((a2.to_ulong() & 0xFF) == b1[3]);

  boost::asio::ip::address_v4 a3(0x01020304);

  assert(a3.to_bytes()[0] == 1);
  assert(a3.to_bytes()[1] == 2);
  assert(a3.to_bytes()[2] == 3);
  assert(a3.to_bytes()[3] == 4);
  assert(a3.to_ulong() == 0x01020304);

  assert(boost::asio::ip::address_v4(0x7F000001).is_loopback());
  assert(boost::asio::ip::address_v4(0x7F000002).is_loopback());
  assert(!boost::asio::ip::address_v4(0x00000000).is_loopback());
  assert(!boost::asio::ip::address_v4(0x01020304).is_loopback());

  assert(boost::asio::ip::address_v4(0x00000000).is_unspecified());
  assert(!boost::asio::ip::address_v4(0x7F000001).is_unspecified());
  assert(!boost::asio::ip::address_v4(0x01020304).is_unspecified());

  assert(boost::asio::ip::address_v4(0x01000000).is_class_a());
  assert(boost::asio::ip::address_v4(0x7F000000).is_class_a());
  assert(!boost::asio::ip::address_v4(0x80000000).is_class_a());
  assert(!boost::asio::ip::address_v4(0xBFFF0000).is_class_a());
  assert(!boost::asio::ip::address_v4(0xC0000000).is_class_a());
  assert(!boost::asio::ip::address_v4(0xDFFFFF00).is_class_a());
  assert(!boost::asio::ip::address_v4(0xE0000000).is_class_a());
  assert(!boost::asio::ip::address_v4(0xEFFFFFFF).is_class_a());
  assert(!boost::asio::ip::address_v4(0xF0000000).is_class_a());
  assert(!boost::asio::ip::address_v4(0xFFFFFFFF).is_class_a());

  assert(!boost::asio::ip::address_v4(0x01000000).is_class_b());
  assert(!boost::asio::ip::address_v4(0x7F000000).is_class_b());
  assert(boost::asio::ip::address_v4(0x80000000).is_class_b());
  assert(boost::asio::ip::address_v4(0xBFFF0000).is_class_b());
  assert(!boost::asio::ip::address_v4(0xC0000000).is_class_b());
  assert(!boost::asio::ip::address_v4(0xDFFFFF00).is_class_b());
  assert(!boost::asio::ip::address_v4(0xE0000000).is_class_b());
  assert(!boost::asio::ip::address_v4(0xEFFFFFFF).is_class_b());
  assert(!boost::asio::ip::address_v4(0xF0000000).is_class_b());
  assert(!boost::asio::ip::address_v4(0xFFFFFFFF).is_class_b());

  assert(!boost::asio::ip::address_v4(0x01000000).is_class_c());
  assert(!boost::asio::ip::address_v4(0x7F000000).is_class_c());
  assert(!boost::asio::ip::address_v4(0x80000000).is_class_c());
  assert(!boost::asio::ip::address_v4(0xBFFF0000).is_class_c());
  assert(boost::asio::ip::address_v4(0xC0000000).is_class_c());
  assert(boost::asio::ip::address_v4(0xDFFFFF00).is_class_c());
  assert(!boost::asio::ip::address_v4(0xE0000000).is_class_c());
  assert(!boost::asio::ip::address_v4(0xEFFFFFFF).is_class_c());
  assert(!boost::asio::ip::address_v4(0xF0000000).is_class_c());
  assert(!boost::asio::ip::address_v4(0xFFFFFFFF).is_class_c());

  assert(!boost::asio::ip::address_v4(0x01000000).is_multicast());
  assert(!boost::asio::ip::address_v4(0x7F000000).is_multicast());
  assert(!boost::asio::ip::address_v4(0x80000000).is_multicast());
  assert(!boost::asio::ip::address_v4(0xBFFF0000).is_multicast());
  assert(!boost::asio::ip::address_v4(0xC0000000).is_multicast());
  assert(!boost::asio::ip::address_v4(0xDFFFFF00).is_multicast());
  assert(boost::asio::ip::address_v4(0xE0000000).is_multicast());
  assert(boost::asio::ip::address_v4(0xEFFFFFFF).is_multicast());
  assert(!boost::asio::ip::address_v4(0xF0000000).is_multicast());
  assert(!boost::asio::ip::address_v4(0xFFFFFFFF).is_multicast());

  boost::asio::ip::address_v4 a4 = boost::asio::ip::address_v4::any();

  assert(a4.to_bytes()[0] == 0);
  assert(a4.to_bytes()[1] == 0);
  assert(a4.to_bytes()[2] == 0);
  assert(a4.to_bytes()[4] == 0);
  assert(a4.to_ulong() == 0);

  boost::asio::ip::address_v4 a5 = boost::asio::ip::address_v4::loopback();

  assert(a5.to_bytes()[0] == 0x7F);
  assert(a5.to_bytes()[1] == 0);
  assert(a5.to_bytes()[2] == 0);
  assert(a5.to_bytes()[3] == 0x01);
  assert(a5.to_ulong() == 0x7F000001);

  boost::asio::ip::address_v4 a6 = boost::asio::ip::address_v4::broadcast();

  assert(a6.to_bytes()[0] == 0xFF);
  assert(a6.to_bytes()[1] == 0xFF);
  assert(a6.to_bytes()[2] == 0xFF);
  assert(a6.to_bytes()[3] == 0xFF);
  assert(a6.to_ulong() == 0xFFFFFFFF);

  boost::asio::ip::address_v4 class_a(0xFF000000);
  boost::asio::ip::address_v4 class_b(0xFFFF0000);
  boost::asio::ip::address_v4 class_c(0xFFFFFF00);
  boost::asio::ip::address_v4 class_x(0xFFFFFFFF);

  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0x01000000)) == class_a);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0x7F000000)) == class_a);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0x80000000)) == class_b);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0xBFFF0000)) == class_b);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0xC0000000)) == class_c);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0xDFFFFF00)) == class_c);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0xE0000000)) == class_x);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0xEFFFFFFF)) == class_x);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0xF0000000)) == class_x);
  assert(boost::asio::ip::address_v4::netmask(
    boost::asio::ip::address_v4(0xFFFFFFFF)) == class_x);
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}

    
