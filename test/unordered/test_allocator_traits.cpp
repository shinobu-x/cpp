#include "implementation.hpp"

#include <boost/limits.hpp>
#include <boost/static_assert.hpp>

#include <cassert>

#define ALLOCATOR_METHODS(name)                                                \
template <typename U>                                                          \
struct rebind {                                                                \
  typedef name<U> other;                                                       \
};                                                                             \
                                                                               \
name() {}                                                                      \
template <typename Y>                                                          \
name (name<Y> const&) {}                                                       \
T* address(T& r) {                                                             \
  return &r;                                                                   \
}                                                                              \
T const* address(T const& r) {                                                 \
  return &r;                                                                   \
}                                                                              \
T* allocate(std::size_t n) {                                                   \
  return static_cast<T*>(::operator new(n * sizeof(T)));                       \
}                                                                              \
T* allocate(std::size_t n, void const*) {                                      \
  return static_cast<T*>(::operator new(n * sizeof(T)));                       \
}                                                                              \
void deallocate(T* p, std::size_t) {                                           \
  ::operator delete((void*)p);                                                 \
}                                                                              \
void construct(T* p, T const& t) {                                             \
  new (p)T(t);                                                                 \
}                                                                              \
void destroy(T* p) {                                                           \
  p->~T();                                                                     \
}                                                                              \
std::size_t max_size() const {                                                 \
  return (std::numeric_limits<std::size_t>::max)();                            \
}                                                                              \
bool operator==(name<T> const&) {                                              \
  return true;                                                                 \
}                                                                              \
bool operator!=(name<T> const&) {                                              \
  return false;                                                                \
}

#define ALLOCATOR_METHODS_TYPEDEFS(name)                                       \
template <typename U>                                                          \
struct rebind {                                                                \
  typedef name<U> other;                                                       \
};                                                                             \
name() {}                                                                      \
template <typename Y>                                                          \
name(name<Y> const&) {}                                                        \
pointer address(T& r) {                                                        \
  return &r;                                                                   \
}                                                                              \
const_pointer address(T const& r) {                                            \
  return &r;                                                                   \
}                                                                              \
pointer allocate(std::size_t n) {                                              \
  return pointer(::operator new(n * sizeof(T)));                               \
}                                                                              \
pointer allocate(std::size_t, void const*) {                                   \
  return pointer(::operator new(n * sizeof(T)));                               \
}                                                                              \
void deallocate(pointer p, std::size_t) {                                      \
  ::operator delete((void*)p);                                                 \
}                                                                              \
void construct(T* p, T const& t) {                                             \
  new (p)T(t);                                                               \
}                                                                              \
void destroy(T* p) {                                                           \
  p->~T();                                                                     \
}                                                                              \
size_type max_size() const                                                     \
  return (std::numeric_limitx<size_type>::max)();                              \
}                                                                              \
bool operator==(name<T> const&) {                                              \
  return true;                                                                 \
}                                                                              \
bool operator!=(name<T> const&) {                                              \
  return false;                                                                \
}

struct yes_type {
  enum {
    value = true
  };
};

struct no_type {
  enum {
    value = false
  };
};

static int selected;
void reset() {
  selected = 0;
}

template <typename allocator_t>
int call_select() {
  typedef boost::unordered::detail::allocator_traits<allocator_t> traits;
  allocator_t alloc;

  reset();

  assert(traits::select_on_container_copy_construction(alloc) == alloc);

  return selected;
}

template <typename T>
struct empty_allocator {
  typedef T value_type;
  ALLOCATOR_METHODS(empty_allocator);
};

void test_empty_allocator() {
  typedef empty_allocator<int> allocator;
  typedef boost::unordered::detail::allocator_traits<allocator> traits;
#if BOOST_UNORDERED_USE_ALLOCATOR_TRAITS == 1
  BOOST_STATIC_ASSERT((boost::is_same<traits::size_type,
    std::make_unsigned<std::ptrdiff_t>::type>::value));
#endif
  BOOST_STATIC_ASSERT(
    (boost::is_same<traits::size_type, std::size_t>::value));
  BOOST_STATIC_ASSERT((boost::is_same<traits::pointer, int*>::value));
  BOOST_STATIC_ASSERT(
    (boost::is_same<traits::const_pointer, int const*>::value));
  BOOST_STATIC_ASSERT((boost::is_same<traits::value_type, int>::value));
  assert(!traits::propagate_on_container_copy_assignment::value);
  assert(!traits::propagate_on_container_move_assignment::value);
  assert(!traits::propagate_on_container_swap::value);
  assert(call_select<allocator>() == 0);
}

auto main() -> decltype(0) {
  return 0;
}

