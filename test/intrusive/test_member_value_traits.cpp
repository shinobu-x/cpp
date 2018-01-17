#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/intrusive/member_value_traits.hpp>

#include <cassert>
#include <iostream>
#include <vector>

struct node_t {
  node_t* prev_;
  node_t* next_;
};

struct node_traits {
  typedef node_t node;
  typedef node_t* node_ptr;
  typedef const node_t* const_node_ptr;
  static node_t* get_next(const node_t* n) {
    return n->next_;
  }
  static void set_next(node_t* n, node_t* next) {
    n->next_ = next;
  }
  static node_t* get_previous(const node* n) {
    return n->prev_;
  }
  static void set_previous(node_t *n, node_t* prev) {
    n->prev_ = prev;
  }
};

class base1 {};
class base2 {};

struct value1 : public base1, public node_t {
  int id_;
  node_t node_;
};

struct value2 : public base2, public node_t {
  int id_;
  node_t node_;
};

typedef boost::intrusive::member_value_traits<
  value1,
  node_traits,
  &value1::node_,
  boost::intrusive::normal_link> value_traits1;
typedef boost::intrusive::member_value_traits<
  value2,
  node_traits,
  &value2::node_,
  boost::intrusive::normal_link> value_traits2;

typedef boost::intrusive::list<
  value1,
  boost::intrusive::value_traits<value_traits1> > value1_list;
typedef boost::intrusive::list<
  value2,
  boost::intrusive::value_traits<value_traits2> > value2_list;

auto main() -> decltype(0) {
  bool result = false;
  typedef std::vector<value1> vector_value1;
  typedef std::vector<value2> vector_value2;

  vector_value1 values1;
  vector_value2 values2;

  for (int i = 0; i < 100; ++i) {
    value1 v1;
    v1.id_ = i;
    values1.push_back(v1);
    value2 v2;
    v2.id_ = i;
    values2.push_back(v2);
  }

  value1_list list1(values1.begin(), values1.end());
  value2_list list2(values2.begin(), values2.end());

  value1_list::const_iterator value1_it = list1.begin();
  value2_list::const_iterator value2_it = list2.begin();

  vector_value1::const_iterator vector_value1_it = values1.begin();
  vector_value2::const_iterator vector_value2_it = values2.begin();

  for (; vector_value1_it != values1.end();
    ++value1_it, ++value2_it, ++vector_value1_it, ++vector_value2_it)
    if (&*value1_it == &*vector_value1_it &&
      &*value2_it == &*vector_value2_it)
        result = true;

  assert(result == true);

  return 0;
}
