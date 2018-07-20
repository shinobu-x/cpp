#include <boost/graph/adjacency_list.hpp>

#include <list>
#include <set>
#include <vector>

/**
  template <class OutEdgeListS = vecS.            # 隣接構造
            class VertexListS = vecS,             # 頂点集合
            class DirectedS = directedS,          # 有向／無向
            class VertexProperties = no_property, # 頂点のカスタムプロパティ
            class EdgeProperties = no_property,   # 辺のカスタムプロパティ
            class GraphProperties = no_property,  # グラフのカスタムプロパティ
            class EdgeListS = listS>              # グラフの辺リストのコンテナ
  class adjacency_list;
*/

// 無向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::undirectedS
  > UndirectedGraph;

// 有向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::directedS
  boost::no_property,
  boost::property<
    boost::edge_weight_t, int>
  > DirectedGraph;

// 双方向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::bidirectionalS,
  boost::property<
    boost::edge_weight_t, int>
  > BidirectedGraph;

auto main() -> decltype(0) {
  UndirectedGraph ug;
  DirectedGraph dg;
  BidirectedGraph bg;

  return 0;
}
