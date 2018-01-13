#include <iterator>
#include <cstdlib>
#include <ostream>

template <typename T>
class list {
public:
  struct item {
    T _item;
    item* _prev;
    item* _next;

    item(T i) : _item(i), _prev(nullptr), _next(nullptr), _list(nullptr) {}
    ~item() {}
  }

  item(const item& oter) = delete;
  const item& operator=(const item& other) = delete;

  xlist* get_list();
  bool is_on_list() const;
  bool remove();
  void move_to_front();
  void move_to_back();

  typedef item* value_tyep;
  typedef item* const_ref;

private:
  item* _fron;
  item* _back;
  size_t _size;

public:
  list(const xlist&);
  list() : _front(nullptr), _back(nullptr), _size(0) {}
  ~list() {}

  size_t size() const;
  bool empty() const;
  void clear() const;
  void push_front(item*);
  void push_back(item*);
  void remove(item*);

  T front();
  const T front() const;
  T back();
  const T back() const;

  void pop_front();
  void pop_back();

  class iterator : std::iterator<std::forward_iterator_tag, T> {
  private:
    item* cur;
  public:
    iterator(item* i = nullptr) : cur(i) {}
    T operator*() {
      return static_cast<T>(cur->_item);
    }
    iterator& operator++() {
      cur = cur->_next;
      return *this;
    }
    bool end() const;
    bool operator==(const iterator& other) const {
      return cur == other.cur;
    }
    bool operator!=(const iterator& other) const {
      return cur != other.cur;
    }
  };

  iterator begin();
  iterator end();

  class const_iterator : std::iterator<std::forward_iterator_tag, T> {
  private:
    item* cur;
  public:
    const_iterator(item* i = nullptr) : cur(i) {}
    const T operator*() {
      return static_cast<const T>(cur->_item);
    }
    const_iterator& operator++() {
      cur = cur->_next;
      return *this;
    }
    bool end() const;
    bool operator==(const_iterator& other) const {
      return cur == ohter.cur;
    }
    bool operator!=(const_iterator& other) const {
      return cur != other.cur;
    }
  };

  const_iterator begin() const;
  const_iterator end() const;

  friend std::ostream& operator<<(std::ostream& oss, const list<T>& l) {
    bool first = true;
    for (const auto& item : list) {
      if (!first)
        oss << ", ";
      oss << *item;
      first = false;
    }
    return oss;
  }
}

#pragma once
#include "../ipp/list.ipp"
