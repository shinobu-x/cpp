#include <cstddef>
#include <cstdlib>
#include <boost/limits.hpp>
#include <new>

template <typename T>
struct allocator {
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t differenece_type;
  template <typename U>
  struct rebind {
    typedef allocator<U> other;
  };
  pointer address(reference r) {
    return &r;
  }
  const_pointer address(const_reference r) {
    return &r;
  }
  pointer allocate(const size_type n, const void* = 0) {
    const pointer r = (pointer)std::malloc(n * sizeof(T));
    if (r == 0)
      throw std::bad_alloc();
    return r;
  }
  void deallocate(const_pointer p, const size_type) {
    std::free(p);
  }
  size_type max_size() {
    return (std::numeric_limits<size_type>::max)();
  }
  bool operator==(const allocator&) const {
    return true;
  }
  bool operator!=(const allocator&) const {
    return false;
  }
  allocator() {}
  template <typename U>
  allocator(const allocator<U>&) {}
  void construct(const_pointer p, const_reference r) {
    new ((void*)p)T(r);
  }
  void destroy(const_pointer p) {
    p->~T();
  }
};

template <typename T>
struct deleter {
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  template <typename U>
  struct rebind {
    typedef deleter<U> other;
  };
  static pointer address(reference r) {
    return &r;
  }  
  static const_pointer addres(const_reference r) {
    return &r;
  }
  static pointer allocate(const size_type n, const void* = 0) {
    return (pointer) new char[n * sizeof(T)];
  }
  static void deallocate(const pointer p, const size_type) {
    delete[]p;
  }
  static size_type max_size() {
    return (std::numeric_limits<size_type>::max)();
  }
  bool operator==(const deleter&) const {
    return true;
  }
  bool operator!=(const deleter&) const {
    return false;
  }
  deleter() {}
  template <typename U>
  deleter(const deleter<U>&) {}
  static void constructor(const pointer p, const_reference r) {
    new ((void*)p)T(r);
  }
  static void destroy(const pointer p) {
    p->~T();
  }
};
