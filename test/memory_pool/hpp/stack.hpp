#include <memory>

template <typename T>
struct node {
  T data;
  node* prev;
};

template <class T, class Alloc = std::allocator<T> >
class stack {
public:
  typedef node<T> node_t;
  typedef typename Alloc::template rebind<node_t>::other allocator;

  stack() : head_(nullptr) {}
  ~stack() { clear(); }

  bool is_empty();
  void clear();
  void push(T);
  T pop();
  T top();

private:
  allocator allocator_;
  node_t* head_;
};

#pragma once
#include "../ipp/stack.ipp"
