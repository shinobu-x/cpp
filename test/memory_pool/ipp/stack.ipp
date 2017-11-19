#include "../hpp/stack.hpp"

template <class T, class Alloc >
bool stack<T, Alloc>::is_empty() {
  return head_ == 0;
}

template <class T, class Alloc>
void stack<T, Alloc>::clear() {
  node_t* curr = head_;
  while (curr) {
    node_t* tmp = curr->prev;
    allocator_.destroy(curr);
    allocator_.deallocate(curr, 1);
    curr = tmp;
  }
  head_ = nullptr;
}

template <class T, class Alloc>
void stack<T, Alloc>::push(T elem) {
  node_t* new_node = allocator_.allocate(1);
  allocator_.construct(new_node, node_t());
  new_node->data = elem;
  new_node->prev = head_;
  head_ = new_node;
}

template <class T, class Alloc>
T stack<T, Alloc>::pop() {
  T result = head_->data;
  node_t* tmp = head_->prev;
  allocator_.destroy(head_);
  allocator_.deallocate(head_, 1);
  head_ = tmp;
  return result;
}

template <class T, class Alloc>
T stack<T, Alloc>::top() {
  return head_->data;
}
