#include <map>
#include <typeinfo>
#include <utility>

class typeinfo {
public:
  typeinfo() : p_(0) {}
  typeinfo(const std::type_info& t) : p_(&t) {}

  inline const char* name() const {
    return p_ ? p_->name() : "";
  }

  inline bool operator< (const typeinfo& that) const {
    return (p_ != that.p_) &&
      (!p_ || (that.p_ && static_cast<bool>(p_->before(*that.p_))));
  }

  inline bool operator== (const typeinfo& that) const {
    return (p_ == that.p_) ||
      (p_ && that.p_ && static_cast<bool>(*p_ == *that.p_));
  }
private:
  const std::type_info* p_;
};

struct virtual_function_table {
  void (*copy)(void*, void*);
  void (*del)(void*);
  void* (*clone)(const void*);
};

template <typename T>
struct local_cast {
  static void copy(void*, void*);
  static void destroy(void*);
  static void* clone(const void*);
};

template <typename userkey_t>
class option_map {
public:
  struct generic_t {
    generic_t();
    generic_t(const generic_t&);
    generic_t& operator= (const generic_t&);
    ~generic_t();
    void* obj;
    const virtual_function_table* vft;
    void (*copy)(void*, const void*);
    void (*del)(void*); 
  };

  template <typename T>
  bool find(const userkey_t&) const;

  bool scan(const userkey_t&) const;

  template <typename T>
  bool get(T&, const userkey_t&) const;

  template <typename T>
  T get(const userkey_t&) const;

  template <typename T>
  bool put(const userkey_t&, const T&);

  size_t size() const;

  ~option_map();

  typedef std::pair<userkey_t, typeinfo> key_t;
  typedef std::map<key_t, generic_t> map_t;
  typedef typename map_t::iterator iterator_t;
  map_t map_;
};

#pragma once
#include "../ipp/virtual_function_table.ipp"
