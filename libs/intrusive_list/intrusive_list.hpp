#ifndef INTRUSIVE_LIST_HPP
#define INTRUSIVE_LIST_HPP

#include <cassert>

template <typename T>
class link_list {
public:
  template <typename R>
  struct node {
    R data;
    node* prev = nullptr;
    node* next = nullptr;
  };

  typedef node<T> data_type;
  data_type* head = nullptr;
  data_type* tail = nullptr;
  data_type* this_node = nullptr;

  link_list() {
    data_type* new_node;
    this_node = new_node;
    add();
  }

  void add() {
    data_type* temp = this_node;;
    if (head == nullptr && tail == nullptr) {
      head = temp;
      tail = temp;
    } else {
      tail = temp;
    }
    temp = nullptr;
  }

  void add(data_type const& new_node) {
    data_type* temp = new_node;
  }
};

#endif
