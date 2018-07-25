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

  std::vector<UGEDescriptor> ug_u_edges;
  std::vector<UGEDescriptor> ug_v_edges;

  ug_u_edges.push_back(boost::add_edge(0, 1, ug_u).first);
  ug_u_edges.push_back(boost::add_edge(0, 2, ug_u).first);
  ug_u_edges.push_back(boost::add_edge(0, 3, ug_u).first);
  ug_u_edges.push_back(boost::add_edge(0, 4, ug_u).first);
  ug_u_edges.push_back(boost::add_edge(1, 2, ug_u).first);
  ug_u_edges.push_back(boost::add_edge(3, 4, ug_u).first);

  ug_v_edges.push_back(boost::add_edge(1, 2, ug_v).first);
  ug_v_edges.push_back(boost::add_edge(2, 0, ug_v).first);
  ug_v_edges.push_back(boost::add_edge(2, 3, ug_v).first);
  ug_v_edges.push_back(boost::add_edge(4, 3, ug_v).first);
  ug_v_edges.push_back(boost::add_edge(0, 3, ug_v).first);
  ug_v_edges.push_back(boost::add_edge(0, 4, ug_v).first);

  return 0;
}
