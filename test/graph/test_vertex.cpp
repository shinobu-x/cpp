#include <boost/graph/adjacency_list.hpp>

#include <iostream>
#include <string>
#include <vector>

enum vertex_location_t {
  vertex_location
};

// include/boost/graph/properties.hpp
namespace boost {
  template <>
  struct property_kind<vertex_location_t> {
    typedef vertex_property_tag type;
  };
}

void test1() {
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
  typedef typename G::vertices_size_type vertices_size_type;

  typedef std::pair<int, int> vertices;

  G g;

  // boost::property<vertex_location_t,
  //                 std::vector<std::string>
  //                >
  location_p l;

  std::pair<int, int> edges[10] = {vertices(/* from */0, /* to */1)};
  boost::add_edge(edges[0].first, edges[0].second, g); // 0 -> 1
  vertices_size_type vertices_size = boost::num_vertices(g);

  boost::property_map<G, vertex_location_t>::type loc_t =
    boost::get(vertex_location_t(), g);

  std::vector<std::string> v1;
  std::vector<std::string> v2;

  v1.push_back("a");
  v1.push_back("A");

  v2.push_back("b");
  v2.push_back("B");

  boost::put(loc_t, 0, v1);
  boost::put(loc_t, 1, v2);

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

  std::cout << s[0] << "\n";
  std::cout << t[0] << "\n";

  std::cout << s[1] << "\n";
  std::cout << t[1] << "\n";

  std::cout << vertices_size << "\n";
}

void test2() {
  typedef boost::property<boost::edge_weight_t,
                          std::pair<std::string,
                                    int> > sub_edge_property;

  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::directedS,
                                sub_edge_property,
                                boost::no_property> sub_graph_t;

  typedef boost::property<vertex_location_t,
                          std::pair<std::string,
                                    std::vector<sub_graph_t> > > location_p;

  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::directedS,
                                location_p,
                                boost::no_property> graph_t;

  typedef typename boost::graph_traits<graph_t>::vertex_descriptor
    vertex_descriptor;
  typedef typename boost::graph_traits<graph_t>::edge_iterator edge_iterator;

  typedef typename boost::property_map<graph_t,
                                       vertex_location_t>::type location_m;
  typedef typename boost::property_traits<location_m>::value_type location_t;
  typedef typename graph_t::vertices_size_type vertices_size_type;

  typedef std::pair<int, int> vertices;

  graph_t g0;
  sub_graph_t sg1;
  sub_graph_t sg2;
  location_t l;

  typedef boost::property<boost::no_property,
                          std::pair<int, int> > sub_location_p;
  typedef typename boost::property_map<sub_graph_t,
                                       boost::no_property> sub_location_m;
  typedef typename boost::property_traits<sub_location_m> sub_location_t;

  std::pair<int, int> edges[10] = { vertices(0, 1) };
  boost::add_edge(edges[0].first, edges[0].second, g0);

  typedef std::pair<std::string,
                   std::vector<sub_graph_t> > details;

  boost::property_map<graph_t, vertex_location_t>::type loc_t =
    boost::get(vertex_location_t(), g0);

  typedef std::vector<sub_graph_t> sg0v;
  sg0v v1;
  sg0v v2;

  v1.push_back(sg1);
  v2.push_back(sg2);

  std::pair<std::string, sg0v> d1 = { details("A", v1) };
  std::pair<std::string, sg0v> d2 = { details("B", v2) };

  boost::put(loc_t, 0, d1);
  boost::put(loc_t, 1, d2);

  location_m loc_m = boost::get(vertex_location_t(), g0);
  std::pair<edge_iterator, edge_iterator> es = boost::edges(g0);

  location_t s = boost::get(loc_m, boost::source(*es.first, g0));
  location_t t = boost::get(loc_m, boost::target(*es.first, g0));

  std::cout << s.first << "\n";
  std::cout << t.first << "\n";

  sub_graph_t g1_sub;
  sub_graph_t g2_sub;

  s.second.push_back(g1_sub);
  t.second.push_back(g2_sub);

  std::pair<int, int> sub_edges[10] = { vertices(0, 1) };
  boost::add_edge(sub_edges[0].first, sub_edges[1].second, g1_sub);
  boost::add_edge(sub_edges[0].first, sub_edges[1].second, g2_sub);

  typedef std::pair<int, int> sub_details;

  std::pair<int, int> sub_d1 = { sub_details(9, 10) };
  std::pair<int, int> sub_d2 = { sub_details(7, 8) };
}

auto main() -> decltype(0) {
  test2();
  return 0;
}
