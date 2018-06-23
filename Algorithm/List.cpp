#include <iostream>

template <typename t>
struct Data;

template <typename T>
struct Node;

Node<Data<int> >* head = nullptr;
Node<Data<int> >* tail = nullptr;

template <typename T>
struct Data {
  T x;
};

template <typename T>
struct Node {
  T data;
  Node* next;
};

template <typename T>
struct List {

  void Add(T& NewNode) {
    if (head == nullptr) {
      head = &NewNode;
      tail = &NewNode;
    } else {
      T* curr = tail;
      tail = &NewNode;
      curr->next = tail;
    }
  }

};

auto main() -> decltype(0) {
  List<Node<Data<int> > > list;
  Node<Data<int> > n1;
  Node<Data<int> > n2;
  Node<Data<int> > n3;

  list.Add(n1);
  list.Add(n2);
  list.Add(n3);

  n1.data.x = 1;
  n2.data.x = 2;
  n3.data.x = 3;

  std::cout << head->data.x << "\n";
  std::cout << head->next->data.x << "\n";
  std::cout << head->next->next->data.x << "\n";
  std::cout << tail->data.x << "\n";

  return 0;
}
