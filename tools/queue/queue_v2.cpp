#include <iostream>
#include <memory>

namespace nonstd {

  template <typename T>
  class queue {
  private:
    struct node {
      std::shared_ptr<T> data_;
      std::unique_ptr<node> next_;
  };
  std::unique_ptr<node> head_;
  node* tail_;

  public:
    queue() : head_(new node), tail_(head_.get()) {}
    queue(const queue& that) = delete;
    queue& operator= (const queue& that) = delete;

    std::shared_ptr<T> pop() {
      if (head_.get() == tail_)
        return std::shared_ptr<T>();

      std::shared_ptr<T> const res(head_->data_);
      std::unique_ptr<node> old_head = std::move(head_);
      head_ = std::move(old_head->next_);

      return res;
    }

    void push(T data) {
      std::shared_ptr<T> _data(std::make_shared<T>(std::move(data)));
      std::unique_ptr<node> _node(new node);

      tail_->data_ = _data;
      node* const _tail = _node.get();
      tail_->next_ = std::move(_node);
      tail_ = _tail;
    }
  };  // End queue
}  // End namespace

template <typename T>
void doit() {
}

auto main() -> decltype(0)
{
  doit<int>();
  return 0;
}
