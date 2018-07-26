#include <boost/graph/adjacency_list.hpp>

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

// 無向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::undirectedS
  > UndirectedGraph;

typedef boost::graph_traits<UndirectedGraph>::vertex_descriptor UGVDescriptor;
typedef boost::graph_traits<UndirectedGraph>::edge_descriptor UGEDescriptor;
typedef boost::graph_traits<UndirectedGraph>::vertex_iterator UGVIterator;
typedef boost::graph_traits<UndirectedGraph>::edge_iterator UGEIterator;

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

// 双方向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::bidirectionalS,
  boost::property<
    boost::edge_weight_t, int>
  > BidirectedGraph;

typedef boost::graph_traits<BidirectedGraph>::vertex_descriptor BGVDescriptor;
typedef boost::graph_traits<BidirectedGraph>::edge_descriptor BGEDescriptor;
typedef boost::graph_traits<BidirectedGraph>::vertex_iterator BGVIterator;
typedef boost::graph_traits<BidirectedGraph>::edge_iterator BGEIterator;

auto main() -> decltype(0) {
  UndirectedGraph ug_u, ug_v;
  DirectedGraph dg_u, dg_v;
  BidirectedGraph bg_u, bg_v;

  const int vertices = 9;
  int capacity[] = {
    10, 20, 20, 20, 40, 40, 20, 20, 20, 10};
  int flow[] = {
    8, 12, 12, 12, 12, 12, 16, 16, 16, 8};

  return 0;
}
