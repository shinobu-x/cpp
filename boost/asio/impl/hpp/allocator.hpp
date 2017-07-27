#include <boost/aligned_storage.hpp>

class allocator {
public:
  allocator()
    : in_use_(false) {}

  void* allocate(std::size_t n) {
    if (in_use_ || n >= 1024)
      return ::operator new(n);
    in_use_ = true;
    return static_cast<void*>(&space_);
  }

  void deallocate(void* p) {
    if (p != static_cast<void*>(&space_))
      ::operator delete(p);
    else
      in_use_ = false;
  }

private:
  allocator(const allocator&);
  allocator& operator= (const allocator&);

  bool in_use_;
  boost::aligned_storage<1024>::type space_;
};
