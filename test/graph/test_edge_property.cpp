#include <boost/graph/adjacency_list.hpp>

#include <iostream>

// Custom edge properties
enum edge_xflow_t {
  edge_xflow
};

enum edge_xcapacity_t {
  edge_xcapacity
};

namespace boost {
  BOOST_INSTALL_PROPERTY(edge, xflow);
  BOOST_INSTALL_PROPERTY(edge, xcapacity);
}

template <typename T>
struct Graph {
   Graph(T& g) : G(g) {}

  // The type for the iterators returned by vertices()
  typedef typename boost::graph_traits<T>::vertex_iterator vertex_iterator;
  // The type for the iterators returned by out_edges()
  typedef typename boost::graph_traits<T>::out_edge_iterator out_edge_iterator;
  // The type for the iterators returned by in_edges()
  typedef typename boost::graph_traits<T>::in_edge_iterator in_edge_iterator;

  typename boost::property_map<T, edge_xcapacity_t>::const_type capacity =
    boost::get(edge_xcapacity, G);
  typename boost::property_map<T, edge_xflow_t>::const_type flow =
    boost::get(edge_xflow, G);

  void operator()(T& g) {
    vertex_iterator it1, end1;
    boost::tie(it1, end1) = boost::vertices(G);

    for (; it1 != end1; ++it1) {

      out_edge_iterator it2, end2;
      // Capacity and flow from source to destination edges
      boost::tie(it2, end2) = boost::out_edges(*it1, G);
      for (; it2 != end2; ++it2) {
        std::cout << "-- [Capacity] " << capacity[*it2] << ", "
          << " [Flow] " << flow[*it2] << " --> "
          << boost::target(*it2, G) << "\t";
      }

      in_edge_iterator it3, end3;
      // Capacity and flow from destination to source edges
      boost::tie(it3, end3) = boost::in_edges(*it1, G);
      for (; it3 != end3; ++it3) {
        std::cout << " <-- " << capacity[*it3] << " [Capacity], "
          << flow[*it3] << " [Flow] " << boost::source(*it3, G) << "\n";
      }
    }

    std::cout << "\n";
  }

  T& G;
};

auto main() -> decltype(0) {

  return 0;
}
