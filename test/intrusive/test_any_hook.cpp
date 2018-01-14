#include <boost/intrusive/any_hook.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/intrusive/list.hpp>

#include <vector>

template <typename T>
class test_t : public boost::intrusive::any_base_hook<> {
private:
  T v_;
public:
  boost::intrusive::any_member_hook<> member_hook_;

  test_t(T i) : v_(i) {}
};

auto main() -> decltype(0) {
  // define a base hook option which converts any_base_hook to a slist hook
  typedef boost::intrusive::any_to_slist_hook<
    boost::intrusive::base_hook<
      boost::intrusive::any_base_hook<> > > base_slist_option;

  typedef boost::intrusive::slist<test_t<int>, base_slist_option> base_slist;

  // define a member hook option which converts any_member_hook to a list hook
  typedef boost::intrusive::any_to_list_hook<
    boost::intrusive::any_to_list_hook<
      boost::intrusive::member_hook<
        test_t<int>,
        boost::intrusive::any_member_hook<>,
        &test_t<int>::member_hook_> > > member_list_option;

  typedef boost::intrusive::list<test_t<int>, member_list_option> member_list;

  // create several test_t instances, each one with a different value
  std::vector<test_t<int> > v;

  for (int i = 0; i < 100; ++i)
    v.push_back(test_t<int>(i));

  base_slist slist;
  member_list mlist;

  auto it = v.begin();
  auto end = v.end();

  // insert them in reverse order in the slist and in order in the list
  for (; it != end; ++it) {
    slist.push_front(*it);
    mlist.push_back(*it);
  }

  auto bit = slist.begin();   // slist
  auto mrit = mlist.rbegin(); // list
  auto rit = v.rbegin();
  auto rend = v.rend();

  // test the objects inserted in the base hook list
  for (; rit != rend; ++rit, ++bit, ++mrit)
    if (&*bit != &*rit ||
        &*mrit != &*rit)
      std::cout << "NG\n";

  return 0;
}
