#include <boost/graph/adjacency_list.hpp>

#include <iostream>
#include <list>
#include <set>
#include <vector>

/**
  template <class OutEdgeListS = vecS.          # 隣接構造
            class VertexListS = vecS,           # 頂点集合
            class DirectedS = directedS,        # 有向／無向
            class VertexProperty = no_property, # 頂点のカスタムプロパティ
            class EdgeProperty = no_property,   # 辺のカスタムプロパティ
            class GraphProperty = no_property,  # グラフのカスタムプロパティ
            class EdgeListS = listS>            # グラフの辺リストのコンテナ
  class adjacency_list;
*/
// 有向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::directedS,
  boost::no_property,
  boost::property<
    boost::edge_weight_t, int>
  > DirectedGraph;

typedef boost::graph_traits<DirectedGraph>::vertex_descriptor DGVDescriptor;
typedef boost::graph_traits<DirectedGraph>::edge_descriptor DGEDescriptor;
typedef boost::graph_traits<DirectedGraph>::vertex_iterator DGVIterator;
typedef boost::graph_traits<DirectedGraph>::edge_descriptor DGEIterator;

// 無向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::undirectedS,
  boost::no_property,
  boost::property<
    boost::edge_weight_t, int>
  > UndirectedGraph;

typedef boost::graph_traits<UndirectedGraph>::vertex_descriptor UGVDescriptor;
typedef boost::graph_traits<UndirectedGraph>::edge_descriptor UGEDescriptor;
typedef boost::graph_traits<UndirectedGraph>::vertex_iterator UGVIterator;
typedef boost::graph_traits<UndirectedGraph>::edge_iterator UGEIterator;

// 双方向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::bidirectionalS,
  boost::no_property,
  boost::property<
    boost::edge_weight_t, int>
  > BidirectedGraph;

typedef boost::graph_traits<BidirectedGraph>::vertex_descriptor BGVDescriptor;
typedef boost::graph_traits<BidirectedGraph>::edge_descriptor BGEDescriptor;
typedef boost::graph_traits<BidirectedGraph>::vertex_iterator BGVIterator;
typedef boost::graph_traits<BidirectedGraph>::edge_iterator BGEIterator;

auto main() -> decltype(0) {
  DirectedGraph dg_u, dg_v;
  UndirectedGraph ug_u, ug_v;
  BidirectedGraph bg_u, bg_v;

  typedef boost::property<boost::edge_name_t, int> edge_property;
  const int vertices = 9;
  int capacity[] = {
    10, 20, 20, 20, 40, 40, 20, 20, 20, 10};
  int flow[] = {
    8, 12, 12, 12, 12, 12, 16, 16, 16, 8};

/**
 // include/boost/graph/detail/adjacency_list.hpp
 Method: add_edge

 O(1) for allow_parallel_edge_tag
 O(log(E/V)) for disallow_parallel_edge_tag

 Directed | Undirected | Bidirected Graphs

 Input: vertex_descriptor,
        vertex_descriptor,
        edge_property_type&,
        directed_graph_helper& |
        undirected_graph_helper& |
        bidirectional_graph_helper_with_property&

 Return Type: std::pair
        Value: edge_descriptor, bool

*/
  boost::add_edge(1, 2, 1, dg_u);
  boost::add_edge(1, 2, 1, ug_u);
  boost::add_edge(1, 2, 1, bg_u);
  return 0;
}
