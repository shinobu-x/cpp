#include <boost/graph/adjacency_list.hpp>

#include <string>
#include <vector>

enum vertex_location_t {
  vertex_location
};

namespace boost {
  BOOST_INSTALL_PROPERTY(vertex, location);
}

auto main() -> decltype(0) {
  typedef boost::property<vertex_location_t,
                          std::vector<std::string> > location;
  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::directedS,
                                location,
                                boost::no_property> G;
  typedef typename boost::property_map<G, vertex_location_t>::type location_t;
  typedef typename boost::graph_traits<G>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<G>::edge_iterator edge_iterator;

  G g;
  location l;

  boost::property_map<G, vertex_location_t>::type loc1 =
    boost::get(vertex_location_t(), g);

  std::vector<std::string> v;
  v.push_back("A");

  boost::put(loc1, 0, v);

  std::pair<edge_iterator, edge_iterator> es = boost::edges(g);
 
  // location_t r = boost::get(vertex_location, g);
  location_t loc2 = boost::get(vertex_location_t(), g);

  return 0;
}
