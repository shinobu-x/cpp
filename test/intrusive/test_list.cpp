#include <boost/intrusive/list.hpp>

#include <cassert>
#include <vector>

template <typename T>
class data : public boost::intrusive::list_base_hook<> {
  T value_;
public:
  boost::intrusive::list_member_hook<> member_hook_;
  data(T value) : value_(value) {} 
};

// base
typedef boost::intrusive::list<data<int> > base_list;
// member
typedef boost::intrusive::list<
  data<int>,
  boost::intrusive::member_hook<
    data<int>,
    boost::intrusive::list_member_hook<>,
    &data<int>::member_hook_> > member_list;

auto main() -> decltype(0) {
  std::vector<data<int> > v;
  for (int i = 0; i < 100; ++i)
    v.push_back(data<int>(i));

  base_list base;
  member_list member;

  for (auto it = v.begin(); it != v.end(); ++it) {
    // base
    base.push_back(*it);
    // member
    member.push_back(*it);
  }

  {
    base_list::reverse_iterator b_r_it = base.rbegin();
    member_list::iterator m_it = member.begin();;
    std::vector<data<int> >::iterator it = v.begin();

    for (; it != v.end(); ++it, ++b_r_it)
      assert(&*b_r_it != &*it);

    for (; it != v.end(); ++it, ++m_it)
      assert(&*m_it != &*it);
  }

  return 0;
}
