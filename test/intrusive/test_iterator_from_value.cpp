#include <boost/intrusive/list.hpp>
#include <boost/intrusive/unordered_set.hpp>

#include <boost/functional/hash.hpp>

#include <cassert>
#include <vector>

template <typename T>
class data {
  T data_;
public:
  void set(T data) {
    data_ = data;
  }

  // member
  boost::intrusive::list_member_hook<> list_hook_;
  boost::intrusive::unordered_set_member_hook<> unordered_set_hook_;

  // operator
  friend bool operator==(const data& a, const data& b) {
    return a.data_ == b.data_;
  }
  friend bool operator!=(const data& a, const data& b) {
    return a.data_ != b.data_;
  }
  
  // hash
  friend std::size_t hash_value(const data& data) {
    return boost::hash<T>()(data.data_);
  }
};

// list
typedef boost::intrusive::member_hook<
  data<int>,
  boost::intrusive::list_member_hook<>,
  &data<int>::list_hook_> list_member;
typedef boost::intrusive::list<
  data<int>,
  list_member> list_t;

// unordered_set
typedef boost::intrusive::member_hook<
  data<int>,
  boost::intrusive::unordered_set_member_hook<>,
  &data<int>::unordered_set_hook_> unordered_set_member;
typedef boost::intrusive::unordered_set<
  data<int>,
  unordered_set_member> unordered_set_t;

auto main() -> decltype(0) {
  std::vector<data<int> > v(100);

  list_t list;
  unordered_set_t::bucket_type buckets[100];
  unordered_set_t unordered_set(
    unordered_set_t::bucket_traits(buckets, 100));

  for (int i = 0; i < 100; ++i)
    v[i].set(i);

  list.insert(list.end(), v.begin(), v.end());
  unordered_set.insert(v.begin(), v.end());

  list_t::iterator list_it = list.begin();
  for (int i = 0; i < 100; ++i, ++list_it)
    assert(list.iterator_to(v[i]) == list_it &&
      list_t::s_iterator_to(v[i]) == list_it);

  unordered_set_t::iterator unordered_set_it = unordered_set.begin();
  for (int i = 0; i < 100; ++i) {
    unordered_set_it = unordered_set.find(v[i]);
    assert(unordered_set.iterator_to(v[i]) == unordered_set_it);

    assert(*unordered_set.local_iterator_to(v[i]) == *unordered_set_it &&
      *unordered_set_t::s_local_iterator_to(v[i]) == *unordered_set_it);
  }

  return 0;
}
