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
struct VertexProperty {
  int Id;
  VertexProperty(int id) : Id(id) {}
};

// 有向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::directedS,
  VertexProperty,
  boost::property<
    boost::edge_weight_t, int>
  > DirectedGraph;

typedef boost::graph_traits<DirectedGraph>::vertex_descriptor
DGVertexDescriptor;
typedef boost::graph_traits<DirectedGraph>::edge_descriptor
DGEdgeescriptor;
typedef boost::graph_traits<DirectedGraph>::vertex_iterator
DGVertexIterator;
typedef boost::graph_traits<DirectedGraph>::edge_descriptor
DGEdgeIterator;

// 無向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::undirectedS,
  VertexProperty,
  boost::property<
    boost::edge_weight_t, int>
  > UndirectedGraph;

typedef boost::graph_traits<UndirectedGraph>::vertex_descriptor
UGVertexDescriptor;
typedef boost::graph_traits<UndirectedGraph>::edge_descriptor
UGEdgeDescriptor;
typedef boost::graph_traits<UndirectedGraph>::vertex_iterator
UGVertexIterator;
typedef boost::graph_traits<UndirectedGraph>::edge_iterator
UGEdgeIterator;

// 双方向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::bidirectionalS,
  VertexProperty,
  boost::property<
    boost::edge_weight_t, int>
  > BidirectedGraph;

typedef boost::graph_traits<BidirectedGraph>::vertex_descriptor
BGVertexDescriptor;
typedef boost::graph_traits<BidirectedGraph>::edge_descriptor
BGEdgeDescriptor;
typedef boost::graph_traits<BidirectedGraph>::vertex_iterator
BGVertexIterator;
typedef boost::graph_traits<BidirectedGraph>::edge_iterator
BGEdgeIterator;

auto main() -> decltype(0) {
  DirectedGraph dg;
  UndirectedGraph ug;
  BidirectedGraph bg;

  {
    auto a = boost::edges(dg);
    auto b = boost::edges(ug);
    auto c = boost::edges(bg);
  }

  {
    int id = 1;
    VertexProperty vp(id);
    auto a = boost::get(&VertexProperty::Id, dg);
    auto b = boost::get(&VertexProperty::Id, ug);
    auto c = boost::get(&VertexProperty::Id, bg);

    DGVertexIterator dg_it;
    auto d = boost::get(&VertexProperty::Id, dg, *dg_it);
    UGVertexIterator ug_it;
    auto e = boost::get(&VertexProperty::Id, ug, *ug_it);
    BGVertexIterator bg_it;
    auto f = boost::get(&VertexProperty::Id, bg, *bg_it);

    DGVertexDescriptor dg_desc;
    auto g = boost::get(&VertexProperty::Id, dg, dg_desc);
    UGVertexDescriptor ug_desc;
    auto h = boost::get(&VertexProperty::Id, ug, ug_desc);
    BGVertexDescriptor bg_desc;
    auto i = boost::get(&VertexProperty::Id, bg, bg_desc);
  }

  return 0;
}
