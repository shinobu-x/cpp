#include <iostream>
#include <vector>

template <typename T>
void doit() {
  typedef std::vector<T> container_type;
  typedef std::vector<container_type> matrix_type;
  typedef container_type value_type;
  typedef container_type ref_type;
  typedef const container_type const_type;
  typedef container_type* ptr_type;
  typedef typename matrix_type::size_type size_type;

  matrix_type matrix;
  container_type detail;

  for (T i=0; i<1000; ++i)
    matrix.push_back(detail);

  for (typename matrix_type::iterator it=matrix.begin();
    it!=matrix.end(); ++it)
    it->push_back(1);

}

auto main() -> decltype(0) 
{
  doit<int>();
  return 0;
}
