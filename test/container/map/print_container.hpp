#include <boost/container/detail/config_begin.hpp>

#include <iostream>

namespace {
  template <class boost_type, class std_type>
  void print_containers(boost_type* boost_t, std_type* std_t) {
    typename boost_type::iterator boost_it = boost_t->begin(), 
      boost_it_end = boost_t->end();
    typename std_type::iterator std_it =std_t->begin(),
      std_it_end = std_t->end();

    std::cout << "boost_type\n";
    for (; boost_it != boost_it_end; ++boost_it)
      std::cout << *boost_it << '\n';

    std::cout << "std_type\n";
    for (; std_it != std_it_end; ++std_it)
      std::cout << *std_it << '\n';
}
} // namespace
