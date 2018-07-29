#include <map> // for vertex_map in copy_impl
#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/operators.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/pending/container_traits.hpp>
#include <boost/range/irange.hpp>
#include <boost/graph/graph_traits.hpp>
#include <memory>
#include <algorithm>
#include <boost/limits.hpp>

#include <boost/iterator/iterator_adaptor.hpp>

#include <boost/mpl/if.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/and.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/pending/container_traits.hpp>
#include <boost/graph/detail/adj_list_edge_iterator.hpp>
#include <boost/graph/properties.hpp>
#include <boost/pending/property.hpp>
#include <boost/graph/adjacency_iterator.hpp>
#include <boost/static_assert.hpp>
#include <boost/assert.hpp>

namespace boost {

  namespace detail {

    // O(E/V)
    template <class EdgeList, class vertex_descriptor>
    void erase_from_incidence_list(EdgeList& el,
      vertex_descriptor v,
      allow_parallel_edge_tag);

    // O(log(E/V))
    template <class EdgeList, class vertex_descriptor>
    void erase_from_incidence_list(EdgeList& el,
      vertex_descriptor v,
      disallow_parallel_edge_tag);

  } // namespace detail

  template <class Tag, class Vertex, class Property>
  const typename property_value<Property,Tag>::type& get(Tag property_tag,
    const detail::stored_edge_property<Vertex, Property>& e);

  template <class Tag, class Vertex, class Iter, class Property>
  const typename property_value<Property,Tag>::type&
  get(Tag property_tag,
    const detail::stored_edge_iter<Vertex, Iter, Property>& e);

  template <class Tag, class Vertex, class EdgeVec, class Property>
  const typename property_value<Property,Tag>::type&
  get(Tag property_tag,
    const detail::stored_ra_edge_iter<Vertex, EdgeVec, Property>& e);

    namespace detail {

      // O(E/V)
      template <class edge_descriptor, class EdgeList, class StoredProperty>
      inline void remove_directed_edge_dispatch(edge_descriptor,
        EdgeList& el,
        StoredProperty& p);

      template <class incidence_iterator, class EdgeList, class Predicate>
      inline void remove_directed_edge_if_dispatch(incidence_iterator first,
        incidence_iterator last,
        EdgeList& el,
        Predicate pred,
        boost::allow_parallel_edge_tag);

      template <class incidence_iterator, class EdgeList, class Predicate>
      inline void remove_directed_edge_if_dispatch(incidence_iterator first,
        incidence_iterator last,
        EdgeList& el,
        Predicate pred,
        boost::disallow_parallel_edge_tag);

      template <class PropT, class Graph, class incidence_iterator,
                class EdgeList, class Predicate>
      inline void undirected_remove_out_edge_if_dispatch(Graph& g,
        incidence_iterator first,
        incidence_iterator last,
        EdgeList& el, Predicate pred,
        boost::allow_parallel_edge_tag);

      template <class PropT, class Graph, class incidence_iterator,
                class EdgeList, class Predicate>
      inline void undirected_remove_out_edge_if_dispatch(Graph& g,
        incidence_iterator first,
        incidence_iterator last,
        EdgeList& el,
        Predicate pred,
        boost::disallow_parallel_edge_tag);

      // O(E/V)
      template <class edge_descriptor, class EdgeList, class StoredProperty>
      inline void remove_directed_edge_dispatch(edge_descriptor e,
        EdgeList& el,
        no_property&)

    } // namespace detail

    // O(1)
    template <class Config>
    inline std::pair<typename Config::edge_iterator, typename Config::edge_iterator>
    edges(const directed_edges_helper<Config>& g_);

    // O(E/V)
    template <class Config>
    inline void remove_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      directed_graph_helper<Config>& g_);

    template <class Config, class Predicate>
    inline void remove_out_edge_if(typename Config::vertex_descriptor u,
      Predicate pred,
      directed_graph_helper<Config>& g_);

    template <class Config, class Predicate>
    inline void remove_edge_if(Predicate pred,
      directed_graph_helper<Config>& g_);

    template <class EdgeOrIter, class Config>
    inline void remove_edge(EdgeOrIter e_or_iter,
      directed_graph_helper<Config>& g_);

    // O(V + E) for allow_parallel_edges
    // O(V * log(E/V)) for disallow_parallel_edges
    template <class Config>
    inline void clear_vertex(typename Config::vertex_descriptor u,
      directed_graph_helper<Config>& g_);

    template <class Config>
    inline void clear_out_edges(typename Config::vertex_descriptor u,
      directed_graph_helper<Config>& g_);

    // O(V)
    template <class Config>
    inline typename Config::edges_size_type
    num_edges(const directed_graph_helper<Config>& g_);

    // O(1) for allow_parallel_edge_tag
    // O(log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline std::pair<typename directed_graph_helper<Config>::edge_descriptor, bool> add_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      const typename Config::edge_property_type& p,
      directed_graph_helper<Config>& g_);

    // Did not use default argument here because that
    // causes Visual C++ to get confused.
    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool> add_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      directed_graph_helper<Config>& g_);

    namespace detail {

      // O(E/V)
      template <class Graph, class EdgeList, class Vertex>
      inline void remove_edge_and_property(Graph& g,
        EdgeList& el,
        Vertex v,
        boost::allow_parallel_edge_tag cat);

      // O(log(E/V))
      template <class Graph, class EdgeList, class Vertex>
      inline void remove_edge_and_property(Graph& g,
        EdgeList& el,
        Vertex v,
        boost::disallow_parallel_edge_tag);

    } // namespace detail

    template <class C>
    inline typename C::InEdgeList& in_edge_list(undirected_graph_helper<C>&,
      typename C::vertex_descriptor v);

    template <class C>
    inline const typename C::InEdgeList& in_edge_list(const undirected_graph_helper<C>&,
      typename C::vertex_descriptor v);

    // O(E/V)
    template <class EdgeOrIter, class Config>
    inline void remove_edge(EdgeOrIter e,
      undirected_graph_helper<Config>& g_);

    // O(E/V) or O(log(E/V))
    template <class Config>
    void remove_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      undirected_graph_helper<Config>& g_);

    template <class Config, class Predicate>
    void remove_out_edge_if(typename Config::vertex_descriptor u,
      Predicate pred,
      undirected_graph_helper<Config>& g_);

    template <class Config, class Predicate>
    void remove_in_edge_if(typename Config::vertex_descriptor u,
      Predicate pred,
      undirected_graph_helper<Config>& g_);

    // O(E/V * E) or O(log(E/V) * E)
    template <class Predicate, class Config>
    void
    remove_edge_if(Predicate pred,
      undirected_graph_helper<Config>& g_);

    // O(1)
    template <class Config>
    inline std::pair<typename Config::edge_iterator, typename Config::edge_iterator> edges(const undirected_graph_helper<Config>& g_);

    // O(1)
    template <class Config>
    inline typename Config::edges_size_type num_edges(const undirected_graph_helper<Config>& g_);

    // O(E/V * E/V)
    template <class Config>
    inline void clear_vertex(typename Config::vertex_descriptor u,
      undirected_graph_helper<Config>& g_);

    // O(1) for allow_parallel_edge_tag
    // O(log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool> add_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      const typename Config::edge_property_type& p,
      undirected_graph_helper<Config>& g_);

    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool> add_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      undirected_graph_helper<Config>& g_);

    // O(1)
    template <class Config>
    inline typename Config::degree_size_type degree(typename Config::vertex_descriptor u,
      const undirected_graph_helper<Config>& g_);

    template <class Config>
    inline std::pair<typename Config::in_edge_iterator, typename Config::in_edge_iterator> in_edges(typename Config::vertex_descriptor u,
      const undirected_graph_helper<Config>& g_);

    template <class Config>
    inline typename Config::degree_size_type in_degree(typename Config::vertex_descriptor u,
      const undirected_graph_helper<Config>& g_);

    template <class C>
    inline typename C::InEdgeList& in_edge_list(bidirectional_graph_helper<C>&,
      typename C::vertex_descriptor v);

    template <class C>
    inline const typename C::InEdgeList& in_edge_list(const bidirectional_graph_helper<C>&,
      typename C::vertex_descriptor v);

    template <class Predicate, class Config>
    inline void remove_edge_if(Predicate pred,
      bidirectional_graph_helper<Config>& g_);

    template <class Config>
    inline std::pair<typename Config::in_edge_iterator, typename Config::in_edge_iterator> in_edges(typename Config::vertex_descriptor u,
      const bidirectional_graph_helper<Config>& g_);

    // O(1)
    template <class Config>
    inline std::pair<typename Config::edge_iterator, typename Config::edge_iterator> edges(const bidirectional_graph_helper<Config>& g_);

    // O(E/V) for allow_parallel_edge_tag
    // O(log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline void remove_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      bidirectional_graph_helper_with_property<Config>& g_);

    // O(E/V) or O(log(E/V))
    template <class EdgeOrIter, class Config>
    inline void remove_edge(EdgeOrIter e,
      bidirectional_graph_helper_with_property<Config>& g_);

    template <class Config, class Predicate>
    inline void remove_out_edge_if(typename Config::vertex_descriptor u,
      Predicate pred,
      bidirectional_graph_helper_with_property<Config>& g_);

    template <class Config, class Predicate>
    inline void remove_in_edge_if(typename Config::vertex_descriptor v,
      Predicate pred,
      bidirectional_graph_helper_with_property<Config>& g_);

    // O(1)
    template <class Config>
    inline typename Config::edges_size_type num_edges(const bidirectional_graph_helper_with_property<Config>& g_);

    // O(E/V * E/V) for allow_parallel_edge_tag
    // O(E/V * log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline void clear_vertex(typename Config::vertex_descriptor u,
      bidirectional_graph_helper_with_property<Config>& g_);

    template <class Config>
    inline void clear_out_edges(typename Config::vertex_descriptor u,
      bidirectional_graph_helper_with_property<Config>& g_);

    template <class Config>
    inline void clear_in_edges(typename Config::vertex_descriptor u,
      bidirectional_graph_helper_with_property<Config>& g_);

    // O(1) for allow_parallel_edge_tag
    // O(log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool> add_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      const typename Config::edge_property_type& p,
      bidirectional_graph_helper_with_property<Config>& g_);

    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool> add_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      bidirectional_graph_helper_with_property<Config>& g_);

    // O(1)
    template <class Config>
    inline typename Config::degree_size_type degree(typename Config::vertex_descriptor u,
      const bidirectional_graph_helper_with_property<Config>& g_);

    template <class Config, class Base>
    inline std::pair<typename Config::adjacency_iterator, typename Config::adjacency_iterator> adjacent_vertices(typename Config::vertex_descriptor u,
      const adj_list_helper<Config, Base>& g_);

    template <class Config, class Base>
    inline std::pair<typename Config::inv_adjacency_iterator, typename Config::inv_adjacency_iterator> inv_adjacent_vertices(typename Config::vertex_descriptor u,
      const adj_list_helper<Config, Base>& g_);

    template <class Config, class Base>
    inline std::pair<typename Config::out_edge_iterator, typename Config::out_edge_iterator> out_edges(typename Config::vertex_descriptor u,
       const adj_list_helper<Config, Base>& g_);

    template <class Config, class Base>
    inline std::pair<typename Config::vertex_iterator, typename Config::vertex_iterator> vertices(const adj_list_helper<Config, Base>& g_);

    template <class Config, class Base>
    inline typename Config::vertices_size_type num_vertices(const adj_list_helper<Config, Base>& g_);

    template <class Config, class Base>
    inline typename Config::degree_size_type out_degree(typename Config::vertex_descriptor u,
      const adj_list_helper<Config, Base>& g_);

    template <class Config, class Base>
    inline std::pair<typename Config::edge_descriptor, bool> edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      const adj_list_helper<Config, Base>& g_);

    template <class Config, class Base>
    inline std::pair<typename Config::out_edge_iterator, typename Config::out_edge_iterator> edge_range(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      const adj_list_helper<Config, Base>& g_);

    template <class Config>
    inline typename Config::degree_size_type in_degree(typename Config::vertex_descriptor u,
      const directed_edges_helper<Config>& g_);

    namespace detail {

      template <class Config, class Base, class Property>
      inline typename boost::property_map<typename Config::graph_type, Property>::type get_dispatch(adj_list_helper<Config,Base>&,
        Property p,
        boost::edge_property_tag);

      template <class Config, class Base, class Property>
      inline typename boost::property_map<typename Config::graph_type, Property>::const_type get_dispatch(const adj_list_helper<Config,Base>&,
        Property p,
        boost::edge_property_tag);

      template <class Config, class Base, class Property>
      inline typename boost::property_map<typename Config::graph_type, Property>::type get_dispatch(adj_list_helper<Config,Base>& g,
        Property p,
        boost::vertex_property_tag);

      template <class Config, class Base, class Property>
      inline typename boost::property_map<typename Config::graph_type, Property>::const_type get_dispatch(const adj_list_helper<Config, Base>& g,
        Property p,
        boost::vertex_property_tag);

    } // namespace detail

    template <class Config, class Base, class Property>
    inline typename boost::property_map<typename Config::graph_type, Property>::type get(Property p,
      adj_list_helper<Config, Base>& g);

    template <class Config, class Base, class Property>
    inline typename boost::property_map<typename Config::graph_type, Property>::const_type get(Property p,
      const adj_list_helper<Config, Base>& g);

    template <class Config, class Base, class Property, class Key>
    inline typename boost::property_traits<typename boost::property_map<typename Config::graph_type, Property>::type >::reference get(Property p,
      adj_list_helper<Config, Base>& g,
      const Key& key);

    template <class Config, class Base, class Property, class Key>
    inline typename boost::property_traits<typename boost::property_map<typename Config::graph_type, Property>::const_type >::reference get(Property p,
      const adj_list_helper<Config, Base>& g,
      const Key& key);

    template <class Config, class Base, class Property, class Key,class Value>
    inline void put(Property p,
      adj_list_helper<Config, Base>& g,
      const Key& key,
      const Value& value);

    // O(1)
    template <class Derived, class Config, class Base>
    inline typename Config::vertex_descriptor add_vertex(adj_list_impl<Derived, Config, Base>& g_);

    // O(1)
    template <class Derived, class Config, class Base>
    inline typename Config::vertex_descriptor add_vertex(const typename Config::vertex_property_type& p,
      adj_list_impl<Derived, Config, Base>& g_);

    // O(1)
    template <class Derived, class Config, class Base>
    inline void remove_vertex(typename Config::vertex_descriptor u,
      adj_list_impl<Derived, Config, Base>& g_);

    // O(V)
    template <class Derived, class Config, class Base>
    inline typename Config::vertex_descriptor vertex(typename Config::vertices_size_type n,
      const adj_list_impl<Derived, Config, Base>& g_);

    namespace detail {

      template <class Graph, class vertex_descriptor>
      inline void remove_vertex_dispatch(Graph& g, vertex_descriptor u,
        boost::directed_tag);

      template <class Graph, class vertex_descriptor>
      inline void remove_vertex_dispatch(Graph& g,
        vertex_descriptor u,
        boost::undirected_tag);

      template <class Graph, class vertex_descriptor>
      inline void remove_vertex_dispatch(Graph& g,
        vertex_descriptor u,
        boost::bidirectional_tag);

      template <class EdgeList, class vertex_descriptor>
      inline void reindex_edge_list(EdgeList& el,
        vertex_descriptor u,
        boost::allow_parallel_edge_tag);

      template <class EdgeList, class vertex_descriptor>
      inline void reindex_edge_list(EdgeList& el,
        vertex_descriptor u,
        boost::disallow_parallel_edge_tag);

    } // namespace detail

    template <class G, class C, class B>
    inline typename C::InEdgeList& in_edge_list(vec_adj_list_impl<G,C,B>& g,
      typename C::vertex_descriptor v);

    template <class G, class C, class B>
    inline const typename C::InEdgeList& in_edge_list(const vec_adj_list_impl<G,C,B>& g,
      typename C::vertex_descriptor v);

    // O(1)
    template <class Graph, class Config, class Base>
    inline typename Config::vertex_descriptor add_vertex(vec_adj_list_impl<Graph, Config, Base>& g_);

    template <class Graph, class Config, class Base>
    inline typename Config::vertex_descriptor add_vertex(const typename Config::vertex_property_type& p,
      vec_adj_list_impl<Graph, Config, Base>& g_);

    template <class Graph, class Config, class Base>
    inline std::pair<typename Config::edge_descriptor, bool> add_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      const typename Config::edge_property_type& p,
      vec_adj_list_impl<Graph, Config, Base>& g_);

    template <class Graph, class Config, class Base>
    inline std::pair<typename Config::edge_descriptor, bool> add_edge(typename Config::vertex_descriptor u,
      typename Config::vertex_descriptor v,
      vec_adj_list_impl<Graph, Config, Base>& g_);

    // O(V + E)
    template <class Graph, class Config, class Base>
    inline void remove_vertex(typename Config::vertex_descriptor v,
      vec_adj_list_impl<Graph, Config, Base>& g_);

    // O(1)
    template <class Graph, class Config, class Base>
    inline typename Config::vertex_descriptor vertex(typename Config::vertices_size_type n,
      const vec_adj_list_impl<Graph, Config, Base>&)

} // namespace boost
