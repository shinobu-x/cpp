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

/**
 * declare the hook with an offset_ptr from boost::interprocess to make
 * this class compatible with shared memory
 */
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

// definition of the shared memory friendly intrusive list
typedef boost::intrusive::list<shared_memory_data<int>> intrusive_list_t;

auto main() -> decltype(0) {
  /**
   * create an intrusive list in shared memory
   * nodes and the container itself must be created in shared memory
   */
  const int elements = 100;
  const int shared_memory_size = 50000;
  const char* shared_memory_name = get_shared_memory_name();

  // erase all old shared memory
  boost::interprocess::shared_memory_object::remove(shared_memory_name);
  boost::interprocess::managed_shared_memory shared_memory(
    boost::interprocess::create_only, shared_memory_name, shared_memory_size);


  // create all nodes in shared memory using a shared memory vector
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

  shared_memory.construct<shared_memory_vector_t>(
    boost::interprocess::anonymous_instance)(
    shared_memory_allocator);

  shared_memory_vector->resize(elements);

  // initialize all the nodes
  for (int i = 0; i < elements; ++i)
    (*shared_memory_vector)[i].set(i);

  // create shared memory intrusive list
  intrusive_list_t* list = shared_memory.construct<intrusive_list_t>(
    boost::interprocess::anonymous_instance)();

  // insert objects stored in shared memory vector in the intrusive list
  list->insert(list->end(),
    shared_memory_vector->begin(), shared_memory_vector->end());

  int checker = 0;

  // check all inserted nodes
  for (intrusive_list_t::const_iterator it = list->begin();
    it != list->end(); ++it)
    if (it->get() != checker)
      return false;

  // delete the list and the nodes
  shared_memory.destroy_ptr(list);
  shared_memory.destroy_ptr(shared_memory_vector);

  boost::interprocess::shared_memory_object::remove(shared_memory_name);

  return 0;
}
