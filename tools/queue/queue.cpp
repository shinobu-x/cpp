#include <iostream>
#include <memory>
#include <vector>

namespace nonstd {

template <typename T>
class queue {
private:
  struct node {
    T data_;
    std::unique_ptr<node> next;
    node(T data) : data_(std::move(data)) {}
  };

  std::unique_ptr<node> head;
  node* tail;

public:
  queue() {}
  queue(const queue& that) = delete;
  queue& operator=(const queue& that) = delete;

  std::shared_ptr<T> try_pop() {
    if (!head)
      return std::shared_ptr<T>();

    std::shared_ptr<T> const res(
      std::make_shared<T>(std::move(head->data)));

    std::unique_ptr<node> const old_head = std::move(head);
    head = std::move(old_head->next);
    return res;
  }

  void push(T new_value) {
    std::unique_ptr<node> p(new node(std::move(new_value)));
    node* const new_tail = p.get();

    if (tail)
      tail->next = std::move(p);
    else
      head = std::move(p);

    tail = new_tail; 
  }
};
} // namespace nonstd
template <typename T>
T doit() {
  T a = 1;
  nonstd::queue<T> obj;
  obj.push(a);

}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
