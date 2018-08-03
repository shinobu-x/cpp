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

  {
    auto a = boost::edges(dg_u);
    auto b = boost::edges(ug_u);
    auto c = boost::edges(bg_u);
  }

  {
    auto a = boost::get(boost::vertex_index, dg_u);
    auto b = boost::get(boost::vertex_index, ug_u);
    auto c = boost::get(boost::vertex_index, bg_u);

    DGVIterator dg_it;
    auto d = boost::get(boost::vertex_index, dg_u, *dg_it);
    UGVIterator ug_it;
    auto e = boost::get(boost::vertex_index, ug_u, *ug_it);
    BGVIterator bg_it;
    auto f = boost::get(boost::vertex_index, bg_u, *bg_it);
  }

  return 0;
}
