#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/iteration_macros.hpp>

#include <boost/type_traits/is_convertible.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

namespace {

template <typename graph_t1, typename graph_t2>
void test_1() {
  graph_t1 g1(24);
  graph_t2 g2(24);

  boost::add_edge(boost::vertex(1, g1), boost::vertex(2, g1), g1);
  boost::add_edge(boost::vertex(1, g2), boost::vertex(2, g2), g2);
  boost::add_edge(boost::vertex(2, g1), boost::vertex(10, g1), g1);
  boost::add_edge(boost::vertex(2, g2), boost::vertex(10, g2), g2);
  boost::add_edge(boost::vertex(2, g1), boost::vertex(5, g1), g1);
  boost::add_edge(boost::vertex(2, g2), boost::vertex(5, g2), g2);
  boost::add_edge(boost::vertex(3, g1), boost::vertex(10, g1), g1);
  boost::add_edge(boost::vertex(3, g2), boost::vertex(10, g2), g2);
  boost::add_edge(boost::vertex(3, g1), boost::vertex(0, g1), g1);
  boost::add_edge(boost::vertex(3, g2), boost::vertex(0, g2), g2);
  boost::add_edge(boost::vertex(4, g1), boost::vertex(5, g1), g1);
  boost::add_edge(boost::vertex(4, g2), boost::vertex(5, g2), g2);
  boost::add_edge(boost::vertex(4, g1), boost::vertex(0, g1), g1);
  boost::add_edge(boost::vertex(4, g2), boost::vertex(0, g2), g2);
  boost::add_edge(boost::vertex(5, g1), boost::vertex(14, g1), g1);
  boost::add_edge(boost::vertex(5, g2), boost::vertex(14, g2), g2);
  boost::add_edge(boost::vertex(6, g1), boost::vertex(3, g1), g1);
  boost::add_edge(boost::vertex(6, g2), boost::vertex(3, g2), g2);
  boost::add_edge(boost::vertex(7, g1), boost::vertex(17, g1), g1);
  boost::add_edge(boost::vertex(7, g2), boost::vertex(17, g2), g2);
  boost::add_edge(boost::vertex(7, g1), boost::vertex(11, g1), g1);
  boost::add_edge(boost::vertex(7, g2), boost::vertex(11, g2), g2);
  boost::add_edge(boost::vertex(8, g1), boost::vertex(17, g1), g1);
  boost::add_edge(boost::vertex(8, g2), boost::vertex(17, g2), g2);
  boost::add_edge(boost::vertex(8, g1), boost::vertex(1, g1), g1);
  boost::add_edge(boost::vertex(8, g2), boost::vertex(1, g2), g2);
  boost::add_edge(boost::vertex(9, g1), boost::vertex(11, g1), g1);
  boost::add_edge(boost::vertex(9, g2), boost::vertex(11, g2), g2);
  boost::add_edge(boost::vertex(9, g1), boost::vertex(1, g1), g1);
  boost::add_edge(boost::vertex(9, g2), boost::vertex(1, g2), g2);
  boost::add_edge(boost::vertex(10, g1), boost::vertex(19, g1), g1);
  boost::add_edge(boost::vertex(10, g2), boost::vertex(19, g2), g2);
  boost::add_edge(boost::vertex(10, g1), boost::vertex(15, g1), g1);
  boost::add_edge(boost::vertex(10, g2), boost::vertex(15, g2), g2);
  boost::add_edge(boost::vertex(10, g1), boost::vertex(8, g1), g1);
  boost::add_edge(boost::vertex(10, g2), boost::vertex(8, g2), g2);
  boost::add_edge(boost::vertex(11, g1), boost::vertex(19, g1), g1);
  boost::add_edge(boost::vertex(11, g2), boost::vertex(19, g2), g2);
  boost::add_edge(boost::vertex(11, g1), boost::vertex(15, g1), g1);
  boost::add_edge(boost::vertex(11, g2), boost::vertex(15, g2), g2);
  boost::add_edge(boost::vertex(11, g1), boost::vertex(4, g1), g1);
  boost::add_edge(boost::vertex(11, g2), boost::vertex(4, g2), g2);
  boost::add_edge(boost::vertex(12, g1), boost::vertex(19, g1), g1);
  boost::add_edge(boost::vertex(12, g2), boost::vertex(19, g2), g2);
  boost::add_edge(boost::vertex(12, g1), boost::vertex(8, g1), g1);
  boost::add_edge(boost::vertex(12, g2), boost::vertex(8, g2), g2);
  boost::add_edge(boost::vertex(12, g1), boost::vertex(4, g1), g1);
  boost::add_edge(boost::vertex(12, g2), boost::vertex(4, g2), g2);
  boost::add_edge(boost::vertex(13, g1), boost::vertex(15, g1), g1);
  boost::add_edge(boost::vertex(13, g2), boost::vertex(15, g2), g2);
  boost::add_edge(boost::vertex(13, g1), boost::vertex(8, g1), g1);
  boost::add_edge(boost::vertex(13, g2), boost::vertex(8, g2), g2);
  boost::add_edge(boost::vertex(13, g1), boost::vertex(4, g1), g1);
  boost::add_edge(boost::vertex(13, g2), boost::vertex(4, g2), g2);
  boost::add_edge(boost::vertex(14, g1), boost::vertex(22, g1), g1);
  boost::add_edge(boost::vertex(14, g2), boost::vertex(22, g2), g2);
  boost::add_edge(boost::vertex(14, g1), boost::vertex(12, g1), g1);
  boost::add_edge(boost::vertex(14, g2), boost::vertex(12, g2), g2);
  boost::add_edge(boost::vertex(15, g1), boost::vertex(22, g1), g1);
  boost::add_edge(boost::vertex(15, g2), boost::vertex(22, g2), g2);
  boost::add_edge(boost::vertex(15, g1), boost::vertex(6, g1), g1);
  boost::add_edge(boost::vertex(15, g2), boost::vertex(6, g2), g2);
  boost::add_edge(boost::vertex(16, g1), boost::vertex(12, g1), g1);
  boost::add_edge(boost::vertex(16, g2), boost::vertex(12, g2), g2);
  boost::add_edge(boost::vertex(16, g1), boost::vertex(6, g1), g1);
  boost::add_edge(boost::vertex(16, g2), boost::vertex(6, g2), g2);
  boost::add_edge(boost::vertex(17, g1), boost::vertex(20, g1), g1);
  boost::add_edge(boost::vertex(17, g2), boost::vertex(20, g2), g2);
  boost::add_edge(boost::vertex(18, g1), boost::vertex(9, g1), g1);
  boost::add_edge(boost::vertex(18, g2), boost::vertex(9, g2), g2);
  boost::add_edge(boost::vertex(19, g1), boost::vertex(23, g1), g1);
  boost::add_edge(boost::vertex(19, g2), boost::vertex(23, g2), g2);
  boost::add_edge(boost::vertex(19, g1), boost::vertex(18, g1), g1);
  boost::add_edge(boost::vertex(19, g2), boost::vertex(18, g2), g2);
  boost::add_edge(boost::vertex(20, g1), boost::vertex(23, g1), g1);
  boost::add_edge(boost::vertex(20, g2), boost::vertex(23, g2), g2);
  boost::add_edge(boost::vertex(20, g1), boost::vertex(13, g1), g1);
  boost::add_edge(boost::vertex(20, g2), boost::vertex(13, g2), g2);
  boost::add_edge(boost::vertex(21, g1), boost::vertex(18, g1), g1);
  boost::add_edge(boost::vertex(21, g2), boost::vertex(18, g2), g2);
  boost::add_edge(boost::vertex(21, g1), boost::vertex(13, g1), g1);
  boost::add_edge(boost::vertex(21, g2), boost::vertex(13, g2), g2);
  boost::add_edge(boost::vertex(22, g1), boost::vertex(21, g1), g1);
  boost::add_edge(boost::vertex(22, g2), boost::vertex(21, g2), g2);
  boost::add_edge(boost::vertex(23, g1), boost::vertex(16, g1), g1);
  boost::add_edge(boost::vertex(23, g2), boost::vertex(16, g2), g2);

  typedef typename boost::property_map<graph_t1, boost::vertex_index_t>::type
    index_map_t1;
  index_map_t1 index_map1 = boost::get(boost::vertex_index_t(), g1);

  typedef typename boost::property_map<graph_t2, boost::vertex_index_t>::type
    index_map_t2;
  index_map_t2 index_map2 = boost::get(boost::vertex_index_t(), g2);

  typename boost::graph_traits<graph_t1>::vertex_iterator vi1, vend1;
  typename boost::graph_traits<graph_t2>::vertex_iterator vi2, vend2;

  typename boost::graph_traits<graph_t1>::adjacency_iterator ai1, aend1;
  typename boost::graph_traits<graph_t2>::adjacency_iterator ai2, aend2;

  for (
    boost::tie(vi1, vend1) = boost::vertices(g1),
    boost::tie(vi2, vend2) = boost::vertices(g2);
    vi1 != vend1;
    ++vi1, ++vi2) {

    assert(boost::get(index_map1, *vi1) ==
      boost::get(index_map2, *vi2));

    for (
     boost::tie(ai1, aend1) = boost::adjacent_vertices(*vi1, g1),
     boost::tie(ai2, aend2) = boost::adjacent_vertices(*vi2, g2);
     ai1 != aend1;
     ++ai1, ++ai2)
     assert(boost::get(index_map1, *ai1) ==
       boost::get(index_map2, *ai2));
  }

  typename boost::graph_traits<graph_t1>::out_edge_iterator oei1, oeend1;
  typename boost::graph_traits<graph_t2>::out_edge_iterator oei2, oeend2;

  for (
    boost::tie(vi1, vend1) = boost::vertices(g1),
    boost::tie(vi2, vend2) = boost::vertices(g2);
    vi1 != vend1;
    ++vi1, ++vi2) {
    assert(boost::get(index_map1, *vi1) ==
      boost::get(index_map2, *vi2));

    for (
      boost::tie(oei1, oeend1) = boost::out_edges(*vi1, g1),
      boost::tie(oei2, oeend2) = boost::out_edges(*vi2, g2);
      oei1 != oeend1;
      ++oei1, ++oei2)
      assert(boost::get(index_map1, boost::target(*oei1, g1)) ==
        boost::get(index_map2, boost::target(*oei2, g2)));
  }

  typename boost::graph_traits<graph_t1>::in_edge_iterator iei1, ieend1;
  typename boost::graph_traits<graph_t2>::in_edge_iterator iei2, ieend2;

  for (
    boost::tie(vi1, vend1) = boost::vertices(g1),
    boost::tie(vi2, vend2) = boost::vertices(g2);
    vi1 != vend1;
    ++vi1, ++vi2) {

    assert(boost::get(index_map1, *vi1) ==
      boost::get(index_map2, *vi2));

    for (
      boost::tie(iei1, ieend1) = boost::in_edges(*vi1, g1),
      boost::tie(iei2, ieend2) = boost::in_edges(*vi2, g2);
      iei1 != ieend1;
      ++iei1, ++iei2)
      assert(boost::get(index_map1, boost::target(*iei1, g1)) ==
        boost::get(index_map2, boost::target(*iei2, g2)));
  }

  std::vector<std::pair<int, int> > edge_pairs_g1;

  BGL_FORALL_EDGES_T(e, g1, graph_t1) {
    edge_pairs_g1.push_back(
      std::make_pair(
        get(index_map1, source(e, g1)),
        get(index_map1, target(e, g1))));
  }

  graph_t2 g3(edge_pairs_g1.begin(), edge_pairs_g1.end(), num_vertices(g1));

  assert(num_vertices(g1) == num_vertices(g3));

  std::vector<std::pair<int, int> > edge_pairs_g3;

  index_map_t2 index_map3 = boost::get(boost::vertex_index_t(), g3);

  BGL_FORALL_EDGES_T(e, g3, graph_t2) {
    edge_pairs_g3.push_back(
      std::make_pair(
        get(index_map3, source(e, g3)),
        get(index_map3, target(e, g3))));
  }

  if (
    boost::is_convertible<
      typename boost::graph_traits<graph_t1>::directed_category*,
      boost::undirected_tag*>::value ||
    boost::is_convertible<
      typename boost::graph_traits<graph_t2>::directed_category*,
      boost::undirected_tag*>::value) {
    for (std::size_t i = 0; i < edge_pairs_g1.size(); ++i)
      if (edge_pairs_g1[i].first < edge_pairs_g1[i].second) 
        std::swap(edge_pairs_g1[i].first, edge_pairs_g1[i].second);

    for (std::size_t i = 0; i < edge_pairs_g3.size(); ++i)
      if (edge_pairs_g3[i].first < edge_pairs_g3[i].second)
        std::swap(edge_pairs_g3[i].first, edge_pairs_g3[i].second);
  }

  std::sort(edge_pairs_g1.begin(), edge_pairs_g1.end());
  std::sort(edge_pairs_g3.begin(), edge_pairs_g3.end());

  edge_pairs_g1.erase(
    std::unique(edge_pairs_g1.begin(), edge_pairs_g1.end()),
    edge_pairs_g1.end());
  edge_pairs_g3.erase(
    std::unique(edge_pairs_g3.begin(), edge_pairs_g3.end()),
    edge_pairs_g3.end());

  assert(edge_pairs_g1 == edge_pairs_g3);
}

template <typename graph_t>
void test_remove_edge() {
  graph_t g(2);
  boost::add_edge(boost::vertex(0, g), boost::vertex(1, g), g);
  assert(boost::num_vertices(g) == 2);
  assert(boost::num_edges(g) == 1);
  boost::remove_edge(boost::vertex(0, g), boost::vertex(1, g), g);
  assert(boost::num_edges(g) == 0);
  boost::remove_edge(boost::vertex(0, g), boost::vertex(1, g), g);
  assert(boost::num_edges(g) == 0);
}

} // namespace

auto main() -> decltype(0) {
  typedef boost::adjacency_list<
    boost::setS, boost::vecS, boost::undirectedS> graph1;
  typedef boost::adjacency_matrix<boost::undirectedS> graph2;

  test_1<graph1, graph2>();
  return 0;
}
