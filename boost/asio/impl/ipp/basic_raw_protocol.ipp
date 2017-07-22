#include "../hpp/basic_raw_protocol.hpp"

template <int V4, int V6, int T, int P>
basic_raw_protocol<V4, V6, T, P> basic_raw_protocol<V4, V6, T, P>::v4() {
  return basic_raw_protocol(P, V4);
}

template <int V4, int V6, int T, int P>
basic_raw_protocol<V4, V6, T, P> basic_raw_protocol<V4, V6, T, P>::v6() {
  return basic_raw_protocol(P, V6);
}

template <int V4, int V6, int T, int P>
int basic_raw_protocol<V4, V6, T, P>::family() const {
  return basic_raw_protocol<V4, V6, T, P>::family_;
}

template <int V4, int V6, int T, int P>
int basic_raw_protocol<V4, V6, T, P>::type() const {
  return basic_raw_protocol<V4, V6, T, P>::type_;
}

template <int V4, int V6, int T, int P>
int basic_raw_protocol<V4, V6, T, P>::protocol() const {
  return basic_raw_protocol<V4, V6, T, P>::protocol_;
}

template <int Domain, int DomainV6, int Type, int Protocol>
basic_raw_protocol<Domain, DomainV6, Type, Protocol>::basic_raw_protocol(
  int protocol, int family)
  : protocol_(protocol), family_(family), type_(Type) {}

