#include "../hpp/option_map.hpp"
template <typename userkey_t>
template <typename T>
bool option_map<userkey_t>::put(const userkey_t& name, const T& value) {
  struct local_cast {
    static void copy(void* dest, const void* src) {
      *static_cast<T*>(dest) = *static_cast<const T*>(src);
    }

    static void destroy(void* p) {
      delete static_cast<T*>(p);
    }
  };

  generic_t& p = map_[key_t(name, typeid(T))];

  p.obj = new T(value);
  p.copy = &local_cast::copy;
  p.del = &local_cast::destroy;

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
