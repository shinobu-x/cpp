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

    std::pair<vertex_iterator, vertex_iterator> v = boost::vertices(G);
    for (; v.first != v.second; ++v.first) {

      // Capacity and flow from source to destination edges
      boost::graph_traits<boost::adjacency_list<T> > oe =
        boost::out_edges(*v.first, G);
      for (; oe.first != oe.second; ++oe.fisrt) {
        std::cout << "-- [Capacity] " << capacity[*oe.first] << ", "
          << " [Flow] " << flow[*oe.first] << " --> "
          << boost::target(*oe.first, G) << "\t";
      }

      // Capacity and flow from destination to source edges
      std::pair<in_edge_iterator, in_edge_iterator> ie =
        boost::in_edges(*v.first, G);
      for (; ie.first != ie.second; ++ie.first) {
        std::cout << " <-- " << capacity[*ie.first] << " [Capacity], "
          << flow[*ie.first] << " [Flow] " << boost::source(*ie.first, G)
          << "\n";
      }
    }

    std::cout << "\n";
  }

  T& G;
};

auto main() -> decltype(0) {
  typedef boost::property<edge_xcapacity_t, int> capacity;
  typedef boost::property<edge_xflow_t, int, capacity> flow;
  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::bidirectionalS,
                                boost::no_property,
                                flow> G;

  const int num_vertices = 9;
  G g(num_vertices);

  /**
   *          e10 <--- e9 <--- e8
   *           |               ^
   *           v               |
   *   e0 ---> e1 ---> e5 ---> e7 ---> e11
   *           |               ^
   *           v               |
   *           e2 ---> e4 ---> e6
   */
  boost::add_edge(0, 1, flow(10, capacity(8)), g);
  boost::add_edge(1, 2, flow(3, capacity(9)), g);
  boost::add_edge(2, 4, flow(5, capacity(10)), g);
  boost::add_edge(4, 6, flow(8, capacity(5)), g);
  boost::add_edge(6, 7, flow(8, capacity(8)), g);
  boost::add_edge(7, 8, flow(10, capacity(12)), g);
  boost::add_edge(8, 9, flow(5, capacity(7)), g);
  boost::add_edge(9, 10, flow(7, capacity(11)), g);
  boost::add_edge(10, 1, flow(9, capacity(6)), g);
  boost::add_edge(1, 5, flow(8, capacity(15)), g);
  boost::add_edge(5, 7, flow(5, capacity(9)), g);
  boost::add_edge(7, 11, flow(10, capacity(9)), g);

  boost::property_map<G, edge_xflow_t>::type f = boost::get(edge_xflow, g);

  boost::graph_traits<G>::vertex_iterator v, v_end;
  boost::graph_traits<G>::out_edge_iterator e, e_end;

  for (boost::tie(v, v_end) = boost::vertices(g); v != v_end; ++v) {
    for (boost::tie(e, e_end) = boost::out_edges(*v, g); e != e_end; ++e) {
    }
  }

  return 0;
}
