#include <climits>
#include <cstddef>

template <typename T, std::size_t block_size = 4096>
class memory_pool {
public:
  typedef T value_type;
  typedef T* ptr_type;
  typedef T& ref_type;
  typedef const T* const_ptr_type;
  typedef const T* const_ref_type;
  typedef std::size_t size_type;
  typedef ptrdiff_t dif_type;
  typedef std::false_type propagate_copy_assignment;
  typedef std::true_type propagate_move_assignment;
  typedef std::true_type propagate_swap;

  template <typename U>
  struct rebind {
    typedef memory_pool<U> other;
  };

  memory_pool() noexcept;
  memory_pool(const memory_pool& that) noexcept;
  memory_pool(memory_pool&& that) noexcept;
  template <class U>
  memory_pool(const memory_pool<U>& that) noexcept;
  ~memory_pool() noexcept;

  memory_pool& operator=(const memory_pool& that) = delete;
  memory_pool& operator=(memory_pool&& memory_pool) noexcept;

  ptr_type address(ref_type o) const noexcept;
  const_ptr_type address(const_ref_type o) const noexcept;

  ptr_type allocate(size_type n = 1, const_ptr_type hint = 0);
  void deallocate(ptr_type p, size_type n = 1);

  size_type max_size() const noexcept;

  template <class U, class... Args>
  void construct(U* p, Args&&... args);
  template <class U>
  void destroy(U* p);

  template <class... Args>
  ptr_type new_elem(Args&&... args);
  void delete_elem(ptr_type p);

private:
  union slot_ {
    value_type elem;
    slot_* next;
  };

  typedef char* data_ptr_;
  typedef slot_ slot_type_;
  typedef slot_* slot_ptr_type_;

  slot_ptr_type_ curr_block_;
  slot_ptr_type_ curr_slot_;
  slot_ptr_type_ last_slot_;
  slot_ptr_type_ free_slot_;

  size_type pad_ptr(data_ptr_ p, size_type align) const noexcept;
  void allocate_block();

  static_assert(block_size >= 2 * sizeof(slot_type_), "blocksize too small.");
};
};
#pragma once
#include "../ipp/memory_pool.ipp"
