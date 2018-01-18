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
  typedef boost::intrusive::circular_list_algorithms<
    data_node_traits<int> > algo;

  data_node<int> node1, node2, node3;
  algo::init_header(&node1);
  assert(algo::count(&node1) == 1);

  // node2 before node1
  algo::link_before(&node1, &node2);
  assert(algo::count(&node1) == 2);

  // node3 after node2
  algo::link_after(&node2, &node3);
  assert(algo::count(&node1) == 3);

  // unlink node3
  algo::unlink(&node3);
  assert(algo::count(&node1) == 2);

  // unlink node2
  algo::unlink(&node2);
  assert(algo::count(&node1) == 1);

  // unlink node1
  algo::unlink(&node1);
  assert(algo::count(&node1) == 1);

  return 0;
}
