#include <boost/intrusive/circular_list_algorithms.hpp>

#include <cassert>

template <typename T>
struct data_node {
  data_node* next;
  data_node* prev;
};

// node traits
template <typename T>
struct data_node_traits {
  // typedef
  typedef data_node<T> node;
  typedef data_node<T>* node_ptr;
  typedef const data_node<T>* const_node_ptr;

  // get
  static node_ptr get_next(const_node_ptr n) {
    return n->next;
  }
  static node_ptr get_previous(const_node_ptr n) {
    return n->prev;
  }
  // set
  static void* set_next(node_ptr n, node_ptr next) {
    n->next = next;
  }
  static void* set_previous(node_ptr n, node_ptr prev) {
    n->prev = prev;
  }
};

auto main() -> decltype(0) {
  return 0;
}
