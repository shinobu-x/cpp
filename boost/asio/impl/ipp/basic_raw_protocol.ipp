#include "../hpp/basic_raw_protocol.hpp"

template <int Family, int FamilyV6, int Type, int Protocol>
static basic_raw_protocol<Family, FamilyV6, Type, Protocol> v4() {
}

template <int Family, int FamilyV6, int Type, int Protocol>
static basic_raw_protocol<Family, FamilyV6, Type, Protocol> v6() {
}

template <int Family, int FamilyV6, int Type, int Protocol>
int family() {
  return Family;
}

template <int Family, int FamilyV6, int Type, int Protocol>
int type() {
  return Type;
}

template <int Family, int FamilyV6, int Type, int Protocol>
int protocol() {
  return Protocol;
}

