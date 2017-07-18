#include "../hpp/virtual_function_table.hpp"

/** local_cast **/
template <typename T>
void local_cast<T>::copy(void* dest, void* src) {
  *static_cast<T*>(dest) = *static_cast<T*>(src);
}

template <typename T>
void local_cast<T>::destroy(void* p) {
  delete static_cast<T*>(p);
}

template <typename T>
void* clone(const void* p) {
  return new T(*static_cast<const T*>(p));
}

/** option_map **/
template <typename userkey_t>
option_map<userkey_t>::generic_t::generic_t() : obj(0), vft(0) {}

template <typename userkey_t>
option_map<userkey_t>::generic_t::generic_t(const generic_t& that)
  : vft(that.vft) {
  if (vft)
    obj = vft->clone(that.obj);
}

template <typename userkey_t>
typename option_map<userkey_t>::generic_t&
option_map<userkey_t>::generic_t::operator= (const generic_t& that) {
  generic_t temp(that);
  std::swap(obj, that.obj);
  swap(vft, that.vft);
  return *this;
}

template <typename userkey_t>
option_map<userkey_t>::generic_t::~generic_t() {
  if (vft && obj)
    (vft->del)(obj);
}

template <typename userkey_t>
template <typename T>
bool option_map<userkey_t>::put(const userkey_t& name, const T& value) {

  static const virtual_function_table vft = {
    &local_cast<T>::copy,
    &local_cast<T>::destroy,
    &local_cast<T>::clone
  };

  generic_t& p = map_[key_t(name, typeid(T))];

  p.obj = new T(value);
  p.table = &vft;
  p.table->copy;
  p.table->del;

  return true;
}

template <typename userkey_t>
size_t option_map<userkey_t>::size() const {
  return map_.size();
}

template <typename userkey_t>
template <typename T>
bool option_map<userkey_t>::find(const userkey_t& name) const {
  return map_.find(key_t(name, typeid(T)) != map_.end());
}

template <typename userkey_t>
bool option_map<userkey_t>::scan(const userkey_t& name) const {
  const typename map_t::const_iterator i =
    map_.upper_bound(key_t(name, typeinfo()));

  return i != map_.end() && i->first.first == name;
}

template <typename userkey_t>
template <typename T>
bool option_map<userkey_t>::get(T& dest, const userkey_t& name) const {
  const typename map_t::const_iterator i = map_.find(key_t(name, typeid(T)));

  const bool test = (i != map_.end());

  if (test && i->second.obj)
    i->second.copy(&dest, i->second.obj);

  return test;
}

template <typename userkey_t>
template <typename T>
T option_map<userkey_t>::get(const userkey_t& name) const {
//  initizlized_value<T> v;
//  get(v.result, name);
//  return v.result;
}

template <typename userkey_t>
option_map<userkey_t>::~option_map() {
  iterator_t i = map_.begin();

  while (i != map_.end()) {
    generic_t& p = (i++)->second;
    if (p.del)
      p.del(p.obj);
  }
}
