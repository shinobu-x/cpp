#include <boost/intrusive/list.hpp>

struct A : public boost::intrusive::list_base_hook<> {
 virtual ~A() {}

};

template<class X>
struct D : A {
 X x;
 D(const X& x) : x(x) {}
};

typedef boost::intrusive::list<A> A_list;


struct data_holder {
 A_list a;

 template<class C>
 static void delete_disposer(C* c){
  delete c;
 }

 template<class X>
 D<X>* insert(const X& x){
  D<X> *t = new D<X>(x);
  a.push_back(*t);
  return t;
 }

 template<class X>
 void remove(D<X>* t){
  A_list::iterator it = A_list::s_iterator_to(*t);
  a.erase_and_dispose(it, delete_disposer<A>);
 }

};

auto main() -> decltype(0) {

 data_holder data;

 D<int> *i = data.insert(10);
 data.remove(i);

 return 0;
}
