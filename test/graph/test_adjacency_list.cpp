#include <boost/graph/adjacency_list.hpp>

#include <iostream>
#include <list>
#include <set>
#include <vector>

/**
  # コンセプト
  template <class G>
  struct GraphConcept {
    * vertex_descriptor      # 頂点の表現方法
    * edge_descriptor        # 辺の表現方法
    * directed_category      # 有向／無向の指定
    * edge_paralell_category # 有向グラフのエッジ追加の際の逆向きエッジの指定
    * traversal_category     # 頂点、エッジへのアクセス方法を指定
    *   # アクセス方法
    *   # 接続グラフ
    *   # 任意の頂点から出るエッジに対するイテレーション
    *   boost::incidence_graph_tag
    *   # 隣接グラフ
    *   # 任意の頂点から出るエッジの先の頂点に対するイテレーション
    *   boost::adjacency_graph_tag
    *   # 双方向グラフ
    *   # 任意の頂点から入るエッジに対するイテレーション
    *   boost::bidirectional_graph_tag
    *   # 頂点リスト
    *   # 全ての頂点に対するイテレーション
    *   boost::vertex_list_graph_tag
    *   # 辺リスト
    *   # 全ての辺に対するイテレーション
    *   boost::edge_list_graph_tag
    *   # 隣接行列
    *   # 2頂点間のエッジの有無の判定
    *   boost::adjacency_matrix_tag
    typedef typename boost::graph_traits<G>::vertex_descriptor
      vertex_descriptor;
    typedef typename boost::graph_traits<G>::edge_descriptor
      edge_descriptor;
    typedef typename boost::graph_traits<G>::directed_category
      directed_category;
    typedef typename boost::graph_traits<G>::edge_parallel_category
      edge_parallel_category;
    typedef typename boost::graph_traits<G>::traversal_category
      traversal_category;

    # Boost concept checks
    void constraints() {
      BOOST_CONCEPT_ASSERT((DefaultConstructibleConcept<vertex_descriptor>));
      BOOST_CONCEPT_ASSERT((EqualityComparableConcept<vertex_descriptor>));
      BOOST_CONCEPT_ASSERT((AssignableConcept<vertex_descriptor>));
      BOOST_CONCEPT_ASSERT((DefaultConstructibleConcept<edge_descriptor>));
      BOOST_CONCEPT_ASSERT((EqualityComparableConcept<edge_descriptor>));
      BOOST_CONCEPT_ASSERT((AssignableConcept<edge_descriptor>));
    }
    G g;
  };
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
