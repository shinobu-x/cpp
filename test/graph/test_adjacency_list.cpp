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

  # adjacency_list
  namespace boost {
    template <class OutEdgeListS = vecS.          # 隣接構造
              class VertexListS = vecS,           # 頂点集合
              class DirectedS = directedS,        # 有向／無向
              class VertexProperty = no_property, # 頂点のカスタムプロパティ
              class EdgeProperty = no_property,   # 辺のカスタムプロパティ
              class GraphProperty = no_property,  # グラフのカスタムプロパティ
              class EdgeListS = listS>            # グラフの辺リストのコンテナ
    class adjacency_list;

    # vecS:      std::vector
    # listS:     std::list
    # slistS:    std::slist
    # setS:      std::set
    # multisetS: std::multiset
    # hash_setS: boost::unordered_set
  }

  # property
  namespace boost {
    template <class PropertyTag,
              class T,
              class NextProperty = no_property>
    struct property;
  }
*/
typedef boost::property<boost::vertex_distance_t,
                        float,
                        boost::property<boost::vertex_name_t,
                                        std::string> > VertexProperty;
typedef boost::property<boost::edge_weight_t,
                        float> EdgeProperty;

struct flow_t {
  typedef boost::edge_property_tag flow;
};

struct capacity_t {
  typedef boost::edge_property_tag capacity;
};

namespace boost {
  enum edge_extraflow_t {
    edge_extra_flow
  };
  enum edge_extracapacity_t {
    edge_extra_capacity
  };
  BOOST_INSTALL_PROPERTY(edge, extraflow);
  BOOST_INSTALL_PROPERTY(edge, extracapacity);
}

typedef boost::property<boost::edge_extracapacity_t,
                        float> ExtraCapacity;
typedef boost::property<boost::edge_extraflow_t,
                        float,
                        ExtraCapacity> ExtraFlow;
typedef boost::adjacency_list<boost::vecS,
                              boost::vecS,
                              boost::no_property,
                              ExtraFlow> G1;

auto main() -> decltype(0) {
  return 0;
}
