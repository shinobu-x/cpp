#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/list.hpp>

#include <iostream>
#include <vector>

// node which will be used with algorithm
struct node_t {
  node_t *prev;
  node_t *next;
};

class test_t1 {};
class test_t2 {};

struct value_t1 : public test_t1, public node_t {
  int v;
};

struct value_t2 : public test_t1, public test_t2, public node_t {
  float v;
};

// define the node traits
struct node_traits_t {
  typedef node_t node;
  typedef node* node_ptr;
  typedef const node* const_node_ptr;

  static node* get_next(const node* n) {
    return n->next;
  }

  static void set_next(node* cur, node* next) {
    cur->next = next;
  }

  static node* get_previous(const node* n) {
    return n->prev;
  }

  static void set_previous(node* cur, node* prev) {
    cur->prev = prev;
  }
};

// templatized value traits for value_t1 and value_t2
template <typename T>
struct value_traits_t {
  typedef node_traits_t node_traits;
  typedef node_traits_t::node_ptr node_ptr;
  typedef node_traits::const_node_ptr const_node_ptr;
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  static const boost::intrusive::link_mode_type link_mode =
    boost::intrusive::normal_link;

  static node_ptr to_node_ptr(value_type& value) {
    return node_ptr(&value);
  }

  static const_node_ptr to_node_ptr(const value_type& value) {
    return const_node_ptr(&value);
  }

  static pointer to_value_ptr(node_ptr n) {
    return static_cast<value_type*>(n);
  }

  static const_pointer to_value_ptr(const_node_ptr n) {
    return static_cast<const value_type*>(n);
  }
};

typedef boost::intrusive::list<value_t1,
  boost::intrusive::value_traits<
    value_traits_t<value_t1> > > list1;

typedef boost::intrusive::list<value_t2,
  boost::intrusive::value_traits<
    value_traits_t<value_t2> > > list2;

auto main() -> decltype(0) {
  typedef std::vector<value_t1> type1;
  typedef std::vector<value_t2> type2;

  // create values with a different internal member
  type1 v1;
  type2 v2;

  for (int i = 0; i < 100; ++i) {
    value_t1 t1;
    t1.v = i;
    v1.push_back(t1);

    value_t2 t2;
    t2.v = (float)i;
    v2.push_back(t2);
  }

  // create the lists with the objects
  list1 l1(v1.begin(), v1.end());
  list2 l2(v2.begin(), v2.end());

  // test both lists
  list1::const_iterator lit1 = l1.begin();
  list1::const_iterator lend1 = l1.end();

  list2::const_iterator lit2 = l2.begin();
  list2::const_iterator lend2 = l2.end();

  type1::const_iterator vit1 = v1.begin();
  type1::const_iterator vend1 = v1.end();

  type2::const_iterator vit2 = v2.begin();
  type2::const_iterator vend2 = v2.end();

  // test objects inserted in lists
  for (; vit1 != vend1; ++vit1, ++lit1, ++vit2, ++lit2)
    if (&*lit1 != &*vit1 ||
        &*lit2 != &*vit2)
      std::cout << "NG\n";

  return 0;
}
