#include <boost/graph/adjacency_list.hpp>

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
// Custom edge properties
struct flow_t {
  typedef boost::edge_property_tag flow;
};
struct capacity_t {
  typedef boost::edge_property_tag capacity;
};

enum edge_extraflow_t {
  edge_extraflow
};
enum edge_extracapacity_t {
  edge_extracapacity
};

// Custom vertex properties
struct name_t {
  typedef boost::vertex_property_tag name;
};

enum vertex_extrainfo_t {
  vertex_extrainfo
};

namespace boost {
  BOOST_INSTALL_PROPERTY(edge, extraflow);
  BOOST_INSTALL_PROPERTY(edge, extracapacity);
  BOOST_INSTALL_PROPERTY(vertex, extrainfo);
}

typedef boost::property<edge_extracapacity_t,
                        float> Capacity_t;
typedef boost::property<edge_extraflow_t,
                        float,
                        Capacity_t> Flow_t;
typedef boost::property<vertex_extrainfo_t,
                        std::string> Vertex_t;

// Custom container
template <class Allocator>
struct list_allocatorS {};

namespace boost {
  template <class Allocator, class ValueType>
  struct container_gen<list_allocatorS<Allocator>, ValueType> {
    typedef typename Allocator::template rebind<ValueType>::other Alloc_t;
    typedef std::list<ValueType, Alloc_t> type;
  };

  template <>
  struct parallel_edge_traits<list_allocatorS<std::allocator<int> > > {
    typedef allow_parallel_edge_tag type;
  };
}

// Overloading push and erase method to add and remove custom container
namespace boost {
  template <class T>
  std::pair<typename list_allocatorS<T>::iterator, bool>
  push(list_allocatorS<T>& c, const T& v) {
    c.push_back(v);
    return std::make_pair(boost::prior(c.end()), true);
  }

  template <class T>
  void erase(list_allocatorS<T>& c, const T& x) {
    c.erase(std::remove(c.begin(), c.end(), x), c.end());
  }
}

void DoIt() {
  {
    typedef boost::adjacency_list<boost::vecS,
                                  boost::vecS,
                                  boost::bidirectionalS,
                                  Vertex_t,
                                  Flow_t> G;
    G g1;
    typename boost::property_map<G, edge_extraflow_t>::type f1 =
      boost::get(edge_extraflow, g1);
    typename boost::property_map<G, edge_extracapacity_t>::type c1 =
      boost::get(edge_extracapacity, g1);
    typename boost::property_map<G, vertex_extrainfo_t>::type v1 =
      boost::get(vertex_extrainfo, g1);

    const G g2;
    typename boost::property_map<G, edge_extraflow_t>::const_type f2 =
      boost::get(edge_extraflow, g2);
    typename boost::property_map<G, edge_extracapacity_t>::const_type c2 =
      boost::get(edge_extracapacity, g2);
    typename boost::property_map<G, vertex_extrainfo_t>::const_type v2 =
      boost::get(vertex_extrainfo, g2);
  }

  {
    typedef boost::adjacency_list<list_allocatorS<std::allocator<int> >,
                                  boost::vecS,
                                  boost::directedS,
                                  Vertex_t,
                                  Flow_t> G;
    G g;
    typename boost::property_map<G, edge_extraflow_t>::type f =
      boost::get(edge_extraflow, g);
    typename boost::property_map<G, edge_extracapacity_t>::type c =
      boost::get(edge_extracapacity, g);
    typename boost::property_map<G, vertex_extrainfo_t>::type v =
      boost::get(vertex_extrainfo, g);
  }
}

auto main() -> decltype(0) {
  return 0;
}
