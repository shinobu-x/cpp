#include <boost/graph/adjacency_list.hpp>

#include <iostream>
#include <string>
#include <vector>

enum extra_vertex_property {
  extra_location
};

// include/boost/graph/properties.hpp
namespace boost {
  template <>
  struct property_kind<extra_vertex_property> {
    typedef vertex_property_tag type;
  };
}

void test1() {
  // Defines custom property
  typedef boost::property<extra_vertex_property,
                          std::vector<std::string> > extra_vertex_p;
  // Defines graph
  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::directedS,
                                extra_vertex_p,
                                boost::no_property> graph_t;
  // Defines property map
  typedef typename boost::property_map<graph_t, extra_vertex_property>::type
    vertex_property_map_t;
  // Defines property traits
  typedef typename boost::property_traits<vertex_property_map_t>::value_type
    vertex_property_traits_t;

  typedef typename boost::graph_traits<graph_t>::edge_iterator edge_iter;
  typedef typename std::pair<int, int> vertices_t;
  typedef typename std::pair<edge_iter, edge_iter> edge_iter_t;

  graph_t g;
  extra_vertex_p property_t;

  vertices_t edges[10] = { vertices_t(0, 1) };
  vertex_property_map_t pm = boost::get(extra_vertex_property(), g);

  std::vector<std::string> v1;
  std::vector<std::string> v2;

  v1.push_back("a");
  v1.push_back("A");
  v2.push_back("b");
  v2.push_back("B");

  boost::put(pm, 0, v1);
  boost::put(pm, 1, v2);

  edge_iter_t es = boost::edges(g);

  vertex_property_traits_t s = boost::get(pm, boost::source(*es.first, g));
  vertex_property_traits_t t = boost::get(pm, boost::target(*es.first, g));
}

void test2() {
  // Defines custom property for sub graphs
  typedef boost::property<extra_vertex_property,
                          std::pair<std::string,
                                    std::string> > s_extra_vertex_p;
  // Defines sub graph
  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::directedS,
                                s_extra_vertex_p,
                                boost::no_property> sub_graph_t;

  // Defines custom property
  typedef boost::property<extra_vertex_property,
                          std::pair<std::string,
                                    std::vector<sub_graph_t> > > extra_vertex_p;
  // Defines graph
  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::directedS,
                                extra_vertex_p,
                                boost::no_property> graph_t;
  // Defines property map
  typedef typename boost::property_map<graph_t, extra_vertex_property>::type
    vertex_property_map_t;
  // Defines property traits
  typedef typename boost::property_traits<vertex_property_map_t>::value_type
    vertex_property_traits_t;

  typedef typename boost::graph_traits<graph_t>::edge_iterator edge_iter;
  typedef typename std::pair<int, int> vertices_t;
  typedef typename std::pair<std::string,
                             std::vector<sub_graph_t> > extra_vertices_t;
  typedef typename std::pair<edge_iter, edge_iter> edge_iter_t;

  graph_t graph;

  vertices_t edges[10] = { vertices_t(0, 1) };
  boost::add_edge(edges[0].first, edges[1].second, graph);
  edge_iter_t es = boost::edges(graph);
}

auto main() -> decltype(0) {
  test1();
  return 0;
}
