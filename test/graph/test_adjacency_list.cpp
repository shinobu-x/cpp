#include <boost/graph/adjacency_list.hpp>

#include <list>
#include <set>
#include <vector>

/**
  template <class OutEdgeListS = vecS.
            class VertexListS = vecS,
            class DirectedS = directedS,
            class VertexProperties = no_property,
            class EdgeProperties = no_property,
            class GraphProperties = no_property,
            class EdgeListS = listS>
  class adjacency_list;
*/
// 無向グラフ
typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::undirectedS
  > UndirectedGraph;

typedef boost::adjacency_list<
  boost::listS,
  boost::vecS,
  boost::directedS
  boost::no_property,
  boost::property<
    boost::edge_weight_t, int>
  > DirectedGraph;

auto main() -> decltype(0) {
  UndirectedGraph ug;
  DirectedGraph dg;

  return 0;
}
