#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

template <typename T>
struct tracker
  : std::allocator<T> {
  typedef typename std::allocator<T>::pointer pointer_type;
  typedef typename std::allocator<T>::size_type size_type;

  template <typename U>
  struct rebind {
    typedef tracker<U> other;
  };

  tracker() {}

  template <typename U>
  tracker(tracker<U> const& u)
    : std::allocator<T>(u) {}

  pointer_type allocate(size_type size,
    std::allocator<void>::const_pointer = 0) {
    void* p = std::malloc(size*sizeof(T));
    if (p == 0)
      throw static_cast<pointer_type>(p);
  }

  void deallocate(pointer_type p, size_type) {
    std::free(p);
  }
};

typedef std::map<void*, std::size_t, std::less<void*>,
  tracker<std::pair<void* const, std::size_t> > > tracker_type;

struct pointer_tracker {
  tracker_type* tracker_t;
  pointer_tracker(tracker_type* tracker) : tracker_t(tracker) {}
  ~pointer_tracker() {
    tracker_type::const_iterator it = tracker_t->begin();
    while (it != tracker_t->end()) {
      std::cerr << "Leaked at " << it->first << ", "
        << it->second << " bytes\n";
      ++it;
    }
  }
};

tracker_type* get_map() {
  static tracker_type* tracker = new (std::malloc(sizeof*tracker))tracker_type;
  static pointer_tracker pointer(tracker);
  return tracker;
}

void* operator new(std::size_t size) throw (std::bad_alloc) {
  void* m = std::malloc(size == 0 ? 1 : size);
  if (m == 0)
    throw std::bad_alloc();
  (*get_map())[m] = size;
  return m;
}

void operator delete(void* m) throw() {
  if (get_map()->erase(m) == 0)
    std::cerr << "Bug: memory at " << m << '\n';
  std::free(m);
}

auto main() -> decltype(0) {
  std::string* s = new std::string;
}
