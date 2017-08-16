#include <boost/function.hpp>

#include <cassert>
#include <functional>

static int alloc_count = 0;
static int dealloc_count = 0;

template <typename T>
struct counting_allocator : public std::allocator<T> {
  template <typename U>
  struct rebind {
    typedef counting_allocator<U> other;
  };

  counting_allocator() {}

  template <typename U>
  counting_allocator(counting_allocator<U>) {}

  T* allocate(std::size_t n) {
    alloc_count++;
    return std::allocator<T>::allocate(n);
  }

  void deallocate(T* p, std::size_t n) {
    dealloc_count++;
    std::allocator<T>::deallocate(p, n);
  }
};

struct enable_small_object_optimization {};

struct disable_small_object_optimization {
  int unused_state_data[32];
};

template <typename base_t>
struct plus : base_t {
  int operator()(int x, int y) const {
    return x + y;
  }
};

static int minus(int x, int y) {
  return x - y;
}

template <typename base_t>
struct nothing : base_t {
  void operator()() const {}
};

static void do_nothing() {}

void reset() {
  alloc_count = 0;
  dealloc_count = 0;
}

auto main() -> int {
  boost::function2<int, int, int> f1;

  f1.assign(plus<disable_small_object_optimization>(),
    counting_allocator<int>());
  f1.clear();

  assert(alloc_count == 1);
  assert(dealloc_count == 1);

  reset();

  assert(alloc_count == 0);
  assert(dealloc_count == 0);

  f1.assign(plus<enable_small_object_optimization>(),
    counting_allocator<int>());
  f1.clear();
  assert(alloc_count == 0);
  assert(dealloc_count == 0);

  f1.assign(plus<disable_small_object_optimization>(), std::allocator<int>());
  f1.clear();
  f1.assign(plus<enable_small_object_optimization>(), std::allocator<int>());
  f1.clear();

  reset();

  f1.assign(&minus, counting_allocator<int>());
  f1.clear();
  assert(alloc_count == 0);
  assert(dealloc_count == 0);
  f1.assign(&minus, std::allocator<int>());
  f1.clear();

  boost::function0<void> f2;
  reset();

  f2.assign(nothing<disable_small_object_optimization>(),
    counting_allocator<int>());
  f2.clear();
  assert(alloc_count == 1);
  assert(dealloc_count == 1);
  alloc_count = 0;
  dealloc_count = 0;

  f2.assign(nothing<enable_small_object_optimization>(),
    counting_allocator<int>());
  f2.clear();
  assert(alloc_count == 0);
  assert(dealloc_count == 0);

  f2.assign(nothing<disable_small_object_optimization>(),
    std::allocator<int>());
  f2.clear();

  f2.assign(nothing<enable_small_object_optimization>(),
    std::allocator<int>());
  f2.clear();

  reset();

  f2.assign(&do_nothing, counting_allocator<int>());
  f2.clear();
  assert(alloc_count == 0);
  assert(dealloc_count == 0);

  boost::function0<void> f3;
  f2.assign(&do_nothing, std::allocator<int>());
  f3.assign(f2, std::allocator<int>());
  
  return 0;
}
