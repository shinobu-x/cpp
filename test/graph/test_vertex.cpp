#include <boost/graph/adjacency_list.hpp>

#include <string>
#include <vector>

enum vertex_location_t {
  vertex_location
};


// namespace boost {
//   BOOST_INSTALL_PROPERTY(vertex, location);
// }

// BOOST_INSTALL_PROPERTY
namespace boost {
  template <>
  struct property_kind<vertex_location_t> {
    typedef vertex_property_tag type;
  };
}

auto main() -> decltype(0) {
  typedef boost::property<vertex_location_t,
                          std::vector<std::string> > location_p;
  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::directedS,
                                location_p,
                                boost::no_property> G;
  typedef typename boost::graph_traits<G>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<G>::edge_iterator edge_iterator;

  typedef typename boost::property_map<G, vertex_location_t>::type location_m;
  typedef typename boost::property_traits<location_m>::value_type location_t;

  G g;
  // boost::property<vertex_location_t,
  //                 std::vector<std::string>
  //                >
  location_p l;

  boost::property_map<G, vertex_location_t>::type loc_t =
    boost::get(vertex_location_t(), g);

  std::vector<std::string> v;
  v.push_back("A");

  boost::put(loc_t, 0, v);

  // boost::property_map<G,
  //                     vertex_location_t
  //                    >::value_type
  location_m loc_m = boost::get(vertex_location_t(), g);
  std::pair<edge_iterator, edge_iterator> es = boost::edges(g);

  // boost::property_traits<boost::property_map<G,
  //                                            vertex_location_t
  //                                           >::type
  //                       >::value_type
  location_t s = boost::get(loc_m, boost::source(*es.first, g));
  location_t t = boost::get(loc_m, boost::target(*es.first, g));

  return 0;
}
