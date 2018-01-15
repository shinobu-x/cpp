#include <boost/intrusive/list.hpp>

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/interprocess/detail/os_thread_functions.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>

#include <iostream>

const char* get_shared_memory_name() {
  std::stringstream s;
  s << "process_"
    << boost::interprocess::ipcdetail::get_current_process_id();
  static std::string str = s.str();
  return str.c_str();
}

template <typename T>
class shared_memory_data : public boost::intrusive::list_base_hook<
  boost::intrusive::void_pointer<
    boost::interprocess::offset_ptr<void> > > {
  T data_;
public:
  T get() const {
    return data_;
  }

  void set(T data) {
    data_ = data;
  }
};

auto main() -> decltype(0) {
  const int elements = 100;
  const int shared_memory_size = 50000;
  const char* shared_memory_name = get_shared_memory_name();

  boost::interprocess::shared_memory_object::remove(shared_memory_name);
  boost::interprocess::managed_shared_memory shared_memory(
    boost::interprocess::create_only, shared_memory_name, shared_memory_size);

  typedef boost::interprocess::allocator<
    shared_memory_data<int>,
    boost::interprocess::managed_shared_memory::segment_manager>
  shared_memory_allocator_t;

  typedef boost::interprocess::vector<
    shared_memory_data<int>,
    shared_memory_allocator_t> shared_memory_vector_t;

  shared_memory_allocator_t shared_memory_allocator(
    shared_memory.get_segment_manager());

  shared_memory_vector_t* shared_memory_vector =
    shared_memory.construct<shared_memory_vector_t>(
      boost::interprocess::anonymous_instance)(shared_memory_allocator);

  return 0;
}
