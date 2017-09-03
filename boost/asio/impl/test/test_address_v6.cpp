#include <boost/asio/ip/address_v6.hpp>

#include <cassert>
#include <sstream>

void test_1() {
  try {
    boost::system::error_code ec;

    boost::asio::ip::address_v6 addr1;
    const boost::asio::ip::address_v6::bytes_type const_bytes_value = {{0}};
    boost::asio::ip::address_v6 addr2(const_bytes_value);

    unsigned long scope_id = addr1.scope_id();
    addr1.scope_id(scope_id);

    assert(addr1.is_unspecified());
    assert(!addr1.is_loopback());
    assert(!addr1.is_multicast());
    assert(!addr1.is_link_local());
    assert(!addr1.is_site_local());
    assert(!addr1.is_v4_mapped());
    assert(!addr1.is_v4_compatible());
    assert(!addr1.is_multicast_node_local());
    assert(!addr1.is_multicast_link_local());
    assert(!addr1.is_multicast_site_local());
    assert(!addr1.is_multicast_org_local());
    assert(!addr1.is_multicast_global());

    boost::asio::ip::address_v6::bytes_type bytes_value = addr1.to_bytes();
    (void)bytes_value;

    std::string string_value = addr1.to_string();
    string_value = addr1.to_string(ec);
    boost::asio::ip::address_v4 addr3 = addr1.to_v4();

    addr1 = boost::asio::ip::address_v6::from_string("0::0");
    addr1 = boost::asio::ip::address_v6::from_string("0::0", ec);
    addr1 = boost::asio::ip::address_v6::from_string(string_value);
    addr1 = boost::asio::ip::address_v6::from_string(string_value, ec);
    addr1 = boost::asio::ip::address_v6::any();
    addr1 = boost::asio::ip::address_v6::loopback();
    addr1 = boost::asio::ip::address_v6::v4_mapped(addr3);
    addr1 = boost::asio::ip::address_v6::v4_compatible(addr3);

    assert(addr1.is_v4_mapped());
    assert(addr1.is_v4_compatible());

    assert(!(addr1 == addr2));
    assert(addr1 != addr2);
    assert(addr1 < addr2);
    assert(!(addr1 > addr2));
    assert(addr1 <= addr2);
    assert(!(addr1 >= addr2));

    std::ostringstream os;
    os << addr1;

#if !defined(BOOST_NO_STD_WSTREAMBUF)
    std::wostringstream wos;
    wos << addr1;
#endif
  } catch (std::exception) {}
}

void test_2() {
  boost::asio::ip::address_v6 a1;
  assert(a1.is_unspecified());
  assert(a1.scope_id() == 0);

  boost::asio::ip::address_v6::bytes_type b1 =
    {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16}};
  boost::asio::ip::address_v6 a2(b1, 12345);

  assert(a2.to_bytes()[0] == 1);
  assert(a2.to_bytes()[1] == 2);
  assert(a2.to_bytes()[2] == 3);
  assert(a2.to_bytes()[3] == 4);
  assert(a2.to_bytes()[4] == 5);
  assert(a2.to_bytes()[5] == 6);
  assert(a2.to_bytes()[6] == 7);
  assert(a2.to_bytes()[7] == 8);
  assert(a2.to_bytes()[8] == 9);
  assert(a2.to_bytes()[9] == 10);
  assert(a2.to_bytes()[10] == 11);
  assert(a2.to_bytes()[11] == 12);
  assert(a2.to_bytes()[12] == 13);
  assert(a2.to_bytes()[13] == 14);
  assert(a2.to_bytes()[14] == 15);
  assert(a2.to_bytes()[15] == 16);
  assert(a2.scope_id() == 12345);

  boost::asio::ip::address_v6 a3;
  a3.scope_id(12345);
  assert(a3.scope_id() == a2.scope_id());

  boost::asio::ip::address_v6 unspecified_address;
  boost::asio::ip::address_v6::bytes_type loopback_bytes = {{0}};
  loopback_bytes[15] = 1;
  boost::asio::ip::address_v6 loopback_address(loopback_bytes);
  boost::asio::ip::address_v6::bytes_type link_local_bytes = {{0xFE, 0x80, 1}};
  boost::asio::ip::address_v6 link_local_address(link_local_bytes);
  boost::asio::ip::address_v6::bytes_type site_local_bytes = {{0xFE, 0xC0, 1}};
  boost::asio::ip::address_v6 site_local_address(site_local_bytes);
  boost::asio::ip::address_v6::bytes_type v4_mapped_bytes = {{0}};
  v4_mapped_bytes[10] = 0xFF, v4_mapped_bytes[11] = 0xFF;
  v4_mapped_bytes[12] = 1, v4_mapped_bytes[13] = 2;
  v4_mapped_bytes[14] = 3, v4_mapped_bytes[15] = 4;
  boost::asio::ip::address_v6 v4_mapped_address(v4_mapped_bytes);
  boost::asio::ip::address_v6::bytes_type v4_compat_bytes = {{0}};
  v4_compat_bytes[12] = 1, v4_compat_bytes[13] = 2;
  v4_compat_bytes[14] = 3, v4_compat_bytes[15] = 4;
  boost::asio::ip::address_v6 v4_compat_address(v4_compat_bytes);
  boost::asio::ip::address_v6::bytes_type mcast_global_bytes =
    {{0xFF, 0x0E, 1}};
  boost::asio::ip::address_v6 mcast_global_address(mcast_global_bytes);
  boost::asio::ip::address_v6::bytes_type mcast_link_local_bytes =
    {{0xFF, 0x02, 1}};
  boost::asio::ip::address_v6 mcast_link_local_address(mcast_link_local_bytes);
  boost::asio::ip::address_v6::bytes_type mcast_node_local_bytes =
    {{0xFF, 0x01, 1}};
  boost::asio::ip::address_v6 mcast_node_local_address(mcast_node_local_bytes);
  boost::asio::ip::address_v6::bytes_type mcast_org_local_bytes =
    {{0xFF, 0x08, 1}};
  boost::asio::ip::address_v6 mcast_org_local_address(mcast_org_local_bytes);
  boost::asio::ip::address_v6::bytes_type mcast_site_local_bytes =
    {{0xFF, 0x05, 1}};
  boost::asio::ip::address_v6 mcast_site_local_address(mcast_site_local_bytes);

  assert(unspecified_address.is_unspecified());
  assert(!loopback_address.is_unspecified());
  assert(!link_local_address.is_unspecified());
  assert(!site_local_address.is_unspecified());
  assert(!v4_mapped_address.is_unspecified());
  assert(!v4_compat_address.is_unspecified());
  assert(!mcast_global_address.is_unspecified());
  assert(!mcast_link_local_address.is_unspecified());
  assert(!mcast_node_local_address.is_unspecified());
  assert(!mcast_org_local_address.is_unspecified());
  assert(!mcast_site_local_address.is_unspecified());

  assert(!unspecified_address.is_loopback());
  assert(loopback_address.is_loopback());
  assert(!link_local_address.is_loopback());
  assert(!site_local_address.is_loopback());
  assert(!v4_mapped_address.is_loopback());
  assert(!v4_compat_address.is_loopback());
  assert(!mcast_global_address.is_loopback());
  assert(!mcast_link_local_address.is_loopback());
  assert(!mcast_node_local_address.is_loopback());
  assert(!mcast_org_local_address.is_loopback());
  assert(!mcast_site_local_address.is_loopback());

  assert(!unspecified_address.is_link_local());
  assert(!loopback_address.is_link_local());
  assert(link_local_address.is_link_local());
  assert(!site_local_address.is_link_local());
  assert(!v4_mapped_address.is_link_local());
  assert(!v4_compat_address.is_link_local());
  assert(!mcast_global_address.is_link_local());
  assert(!mcast_link_local_address.is_link_local());
  assert(!mcast_node_local_address.is_link_local());
  assert(!mcast_org_local_address.is_link_local());
  assert(!mcast_site_local_address.is_link_local());

  assert(!unspecified_address.is_site_local());
  assert(!loopback_address.is_site_local());
  assert(!link_local_address.is_site_local());
  assert(site_local_address.is_site_local());
  assert(!v4_mapped_address.is_site_local());
  assert(!v4_compat_address.is_site_local());
  assert(!mcast_global_address.is_site_local());
  assert(!mcast_link_local_address.is_site_local());
  assert(!mcast_node_local_address.is_site_local());
  assert(!mcast_org_local_address.is_site_local());
  assert(!mcast_site_local_address.is_site_local());

  assert(!unspecified_address.is_v4_mapped());
  assert(!loopback_address.is_v4_mapped());
  assert(!link_local_address.is_v4_mapped());
  assert(v4_mapped_address.is_v4_mapped());
  assert(!v4_compat_address.is_v4_mapped());
  assert(!mcast_global_address.is_v4_mapped());
  assert(!mcast_link_local_address.is_v4_mapped());
  assert(!mcast_node_local_address.is_v4_mapped());
  assert(!mcast_org_local_address.is_v4_mapped());
  assert(!mcast_site_local_address.is_v4_mapped());

  assert(!unspecified_address.is_v4_compatible());
  assert(!loopback_address.is_v4_compatible());
  assert(!link_local_address.is_v4_compatible());
  assert(!v4_mapped_address.is_v4_compatible());
  assert(v4_compat_address.is_v4_compatible());
  assert(!mcast_global_address.is_v4_compatible());
  assert(!mcast_link_local_address.is_v4_compatible());
  assert(!mcast_node_local_address.is_v4_compatible());
  assert(!mcast_org_local_address.is_v4_compatible());
  assert(!mcast_site_local_address.is_v4_compatible());

  assert(!unspecified_address.is_multicast());
  assert(!loopback_address.is_multicast());
  assert(!link_local_address.is_multicast());
  assert(!v4_mapped_address.is_multicast());
  assert(!v4_compat_address.is_multicast());
  assert(mcast_global_address.is_multicast());
  assert(mcast_link_local_address.is_multicast());
  assert(mcast_node_local_address.is_multicast());
  assert(mcast_org_local_address.is_multicast());
  assert(mcast_site_local_address.is_multicast());

  assert(!unspecified_address.is_multicast_global());
  assert(!loopback_address.is_multicast_global());
  assert(!link_local_address.is_multicast_global());
  assert(!v4_mapped_address.is_multicast_global());
  assert(!v4_compat_address.is_multicast_global());
  assert(mcast_global_address.is_multicast_global());
  assert(!mcast_link_local_address.is_multicast_global());
  assert(!mcast_node_local_address.is_multicast_global());
  assert(!mcast_org_local_address.is_multicast_global());
  assert(!mcast_site_local_address.is_multicast_global());

  assert(!unspecified_address.is_multicast_link_local());
  assert(!loopback_address.is_multicast_link_local());
  assert(!link_local_address.is_multicast_link_local());
  assert(!v4_mapped_address.is_multicast_link_local());
  assert(!v4_compat_address.is_multicast_link_local());
  assert(!mcast_global_address.is_multicast_link_local());
  assert(mcast_link_local_address.is_multicast_link_local());
  assert(!mcast_node_local_address.is_multicast_link_local());
  assert(!mcast_org_local_address.is_multicast_link_local());
  assert(!mcast_site_local_address.is_multicast_link_local());

  assert(!unspecified_address.is_multicast_node_local());
  assert(!loopback_address.is_multicast_node_local());
  assert(!link_local_address.is_multicast_node_local());
  assert(!v4_mapped_address.is_multicast_node_local());
  assert(!v4_compat_address.is_multicast_node_local());
  assert(!mcast_global_address.is_multicast_node_local());
  assert(!mcast_link_local_address.is_multicast_node_local());
  assert(mcast_node_local_address.is_multicast_node_local());
  assert(!mcast_org_local_address.is_multicast_node_local());
  assert(!mcast_site_local_address.is_multicast_node_local());

  assert(!unspecified_address.is_multicast_org_local());
  assert(!loopback_address.is_multicast_org_local());
  assert(!link_local_address.is_multicast_org_local());
  assert(!v4_mapped_address.is_multicast_org_local());
  assert(!v4_compat_address.is_multicast_org_local());
  assert(!mcast_global_address.is_multicast_org_local());
  assert(!mcast_link_local_address.is_multicast_org_local());
  assert(!mcast_node_local_address.is_multicast_org_local());
  assert(mcast_org_local_address.is_multicast_org_local());
  assert(!mcast_site_local_address.is_multicast_org_local());

  assert(!unspecified_address.is_multicast_site_local());
  assert(!loopback_address.is_multicast_site_local());
  assert(!link_local_address.is_multicast_site_local());
  assert(!v4_mapped_address.is_multicast_site_local());
  assert(!v4_compat_address.is_multicast_site_local());
  assert(!mcast_global_address.is_multicast_site_local());
  assert(!mcast_link_local_address.is_multicast_site_local());
  assert(!mcast_node_local_address.is_multicast_site_local());
  assert(!mcast_org_local_address.is_multicast_site_local());
  assert(mcast_site_local_address.is_multicast_site_local());

  assert(boost::asio::ip::address_v6::loopback().is_loopback());
}

auto main() -> decltype(0) {
  test_1(); test_2();
  return 0;
}
