#include <boost/asio/ip/address.hpp>

#include <sstream>

void test_1() {
  try {
    boost::system::error_code ec;

    boost::asio::ip::address addr1;
    const boost::asio::ip::address_v4 const_addr_v4;
    boost::asio::ip::address addr2;
    const boost::asio::ip::address_v6 const_addr_v6;
    boost::asio::ip::address addr3(const_addr_v6);

    bool b = addr1.is_v4();
    (void)b;

    b = addr1.is_loopback();
    (void)b;

    b = addr1.is_unspecified();
    (void)b;

    b = addr1.is_multicast();
    (void)b;

    boost::asio::ip::address_v4 addr_v4_value = addr1.to_v4();
    (void)addr_v4_value;

    boost::asio::ip::address_v6 addr_v6_value = addr1.to_v6();
    (void)addr_v6_value;

    std::string string_value = addr1.to_string();
    string_value = addr1.to_string(ec);

    addr1 = boost::asio::ip::address::from_string("127.0.0.1");
    addr1 = boost::asio::ip::address::from_string("127.0.0.1", ec);
    addr1 = boost::asio::ip::address::from_string(string_value);
    addr1 = boost::asio::ip::address::from_string(string_value, ec);

    b = (addr1 == addr2);
    (void)b;

    b = (addr1 != addr2);
    (void)b;

    b = (addr1 < addr2);
    (void)b;

    b = (addr1 > addr2);
    (void)b;

    b = (addr1 <= addr2);
    (void)b;

    b = (addr1 >= addr2);
    (void)b;

    std::ostringstream os;
    os << addr1;

#if !defined(BOOST_NO_STD_WSTREAMBUF)
    std::wostringstream wos;
    wos << addr1;
#endif
  } catch (std::exception&) {
  }
}

auto main() -> decltype(0) {
  test_1();
  return 0;
} 
