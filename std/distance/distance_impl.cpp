#include <iterator>
#include <iostream>
#include <vector>
#include <list>

#define M_NS_BEGIN(x) namespace x {
#define M_NS_END      }

M_NS_BEGIN(nonstd);
template <typename InputIterator>
typename std::iterator_traits<InputIterator>::difference_type distance_impl(
  InputIterator first,
  InputIterator last,
  std::input_iterator_tag) {
  using result_type =
    typename std::iterator_traits<InputIterator>::difference_type;

  result_type n = 0;

  for (; first != last; ++first) {
    ++n;
  }

  return n;
}

template <class RandomAccessIterator>
typename std::iterator_traits<RandomAccessIterator>::difference_type
  distance_imple(
    RandomAccessIterator first,
    RandomAccessIterator last,
    std::random_access_iterator_tag) {
    return last -first;
}

template <class InputIterator>
typename std::iterator_traits<InputIterator>::difference_type distance(
  InputIterator first,
  InputIterator last) {
  return distance_impl(first, last,
    typename std::iterator_traits<InputIterator>::iterator_category());
}
M_NS_END;

template <typename T>
T doit() {
  std::vector<T> v = {4, 3, 2, 1};
  std::cout << nonstd::distance(v.begin(), v.end()) << '\n';
}
auto main() -> int
{
  doit<int>();
  return 0;
} 
