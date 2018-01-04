#include <iostream>
#include <vector>

#include <boost/lambda/lambda.hpp>
#include <boost/range/algorithm/find_if.hpp>

struct doit {
  int id_;

  doit(int id) : id_(id) {}
};

auto main() -> decltype(0) {
  std::vector<doit> v;
  for (int i = 0; i < 10000; ++i)
    v.push_back(doit(i));

  std::vector<doit>::iterator it = boost::find_if(v,
    &boost::lambda::_1->* &doit::id_ == 3);

  if (it != v.end())
    std::cout << it->id_ << '\n';
  else
    std::cout << '\n';

  return 0;
}
