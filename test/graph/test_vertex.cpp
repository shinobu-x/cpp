#include <boost/graph/adjacency_list.hpp>

#include <iostream>

enum vertex_location_t {
  vertex_location
};

namespace boost {
  BOOST_INSTALL_PROPERTY(vertex, location);
}

auto main() -> decltype(0) {
  typedef boost::property<vertex_location_t, std::string> location;
  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::directedS,
                                location,
                                boost::no_property> G;
  G g;

  typedef typename boost::graph_traits<G>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<G>::edge_iterator edge_iterator;
  location l;

  boost::property_map<G, vertex_location_t>::type loc =
    boost::get(vertex_location_t(), g);

  boost::put(loc, 0, "A");

  std::pair<edge_iterator, edge_iterator> es = boost::edges(g);

  

  return 0;
}
