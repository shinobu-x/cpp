#include <boost/graph/adjacency_list.hpp>

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

  vertex_descriptor v = boost::add_vertex(g);

  return 0;
}
