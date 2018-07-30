namespace boost {

  namespace detail {

    // O(E/V)
    template <class EdgeList, class vertex_descriptor>
    void erase_from_incidence_list(EdgeList& el, vertex_descriptor v,
                                   allow_parallel_edge_tag)
    {
      boost::graph_detail::erase_if(el, detail::target_is<vertex_descriptor>(v));
    }

    // O(log(E/V))
    template <class EdgeList, class vertex_descriptor>
    void erase_from_incidence_list(EdgeList& el, vertex_descriptor v,
                                   disallow_parallel_edge_tag)
    {
      typedef typename EdgeList::value_type StoredEdge;
      el.erase(StoredEdge(v));
    }

    template <class Vertex>
    no_property stored_edge<Vertex>::s_prop;

  } // namespace detail

  template <class Tag, class Vertex, class Property>
  const typename property_value<Property,Tag>::type&
  get(Tag property_tag,
      const detail::stored_edge_property<Vertex, Property>& e)
  {
    return get_property_value(e.get_property(), property_tag);
  }

  template <class Tag, class Vertex, class Iter, class Property>
  const typename property_value<Property,Tag>::type&
  get(Tag property_tag,
      const detail::stored_edge_iter<Vertex, Iter, Property>& e)
  {
    return get_property_value(e.get_property(), property_tag);
  }

  template <class Tag, class Vertex, class EdgeVec, class Property>
  const typename property_value<Property,Tag>::type&
  get(Tag property_tag,
      const detail::stored_ra_edge_iter<Vertex, EdgeVec, Property>& e)
  {
    return get_property_value(e.get_property(), property_tag);
  }

    //=========================================================================
    // Directed Edges Helper Class

    namespace detail {

      // O(E/V)
      template <class edge_descriptor, class EdgeList, class StoredProperty>
      inline void
      remove_directed_edge_dispatch(edge_descriptor, EdgeList& el,
                                    StoredProperty& p)
      {
        for (typename EdgeList::iterator i = el.begin();
             i != el.end(); ++i)
          if (&(*i).get_property() == &p) {
            el.erase(i);
            return;
          }
      }

      template <class incidence_iterator, class EdgeList, class Predicate>
      inline void
      remove_directed_edge_if_dispatch(incidence_iterator first,
                                       incidence_iterator last,
                                       EdgeList& el, Predicate pred,
                                       boost::allow_parallel_edge_tag)
      {
        // remove_if
        while (first != last && !pred(*first))
          ++first;
        incidence_iterator i = first;
        if (first != last)
          for (++i; i != last; ++i)
            if (!pred(*i)) {
              *first.base() = BOOST_GRAPH_MOVE_IF_POSSIBLE(*i.base());
              ++first;
            }
        el.erase(first.base(), el.end());
      }
      template <class incidence_iterator, class EdgeList, class Predicate>
      inline void
      remove_directed_edge_if_dispatch(incidence_iterator first,
                                       incidence_iterator last,
                                       EdgeList& el,
                                       Predicate pred,
                                       boost::disallow_parallel_edge_tag)
      {
        for (incidence_iterator next = first;
             first != last; first = next) {
          ++next;
          if (pred(*first))
            el.erase( first.base() );
        }
      }

      template <class PropT, class Graph, class incidence_iterator,
                class EdgeList, class Predicate>
      inline void
      undirected_remove_out_edge_if_dispatch(Graph& g,
                                             incidence_iterator first,
                                             incidence_iterator last,
                                             EdgeList& el, Predicate pred,
                                             boost::allow_parallel_edge_tag)
      {
        typedef typename Graph::global_edgelist_selector EdgeListS;
        BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

        // remove_if
        while (first != last && !pred(*first))
          ++first;
        incidence_iterator i = first;
        bool self_loop_removed = false;
        if (first != last)
          for (; i != last; ++i) {
            if (self_loop_removed) {
              /* With self loops, the descriptor will show up
               * twice. The first time it will be removed, and now it
               * will be skipped.
               */
              self_loop_removed = false;
            }
            else if (!pred(*i)) {
              *first.base() = BOOST_GRAPH_MOVE_IF_POSSIBLE(*i.base());
              ++first;
            } else {
              if (source(*i, g) == target(*i, g)) self_loop_removed = true;
              else {
                // Remove the edge from the target
                detail::remove_directed_edge_dispatch
                  (*i,
                   g.out_edge_list(target(*i, g)),
                   *(PropT*)(*i).get_property());
              }

              // Erase the edge property
              g.m_edges.erase( (*i.base()).get_iter() );
            }
          }
        el.erase(first.base(), el.end());
      }
      template <class PropT, class Graph, class incidence_iterator,
                class EdgeList, class Predicate>
      inline void
      undirected_remove_out_edge_if_dispatch(Graph& g,
                                             incidence_iterator first,
                                             incidence_iterator last,
                                             EdgeList& el,
                                             Predicate pred,
                                             boost::disallow_parallel_edge_tag)
      {
        typedef typename Graph::global_edgelist_selector EdgeListS;
        BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

        for (incidence_iterator next = first;
             first != last; first = next) {
          ++next;
          if (pred(*first)) {
            if (source(*first, g) != target(*first, g)) {
              // Remove the edge from the target
              detail::remove_directed_edge_dispatch
                (*first,
                 g.out_edge_list(target(*first, g)),
                 *(PropT*)(*first).get_property());
            }

            // Erase the edge property
            g.m_edges.erase( (*first.base()).get_iter() );

            // Erase the edge in the source
            el.erase( first.base() );
          }
        }
      }

      // O(E/V)
      template <class edge_descriptor, class EdgeList, class StoredProperty>
      inline void
      remove_directed_edge_dispatch(edge_descriptor e, EdgeList& el,
                                    no_property&)
      {
        for (typename EdgeList::iterator i = el.begin();
             i != el.end(); ++i)
          if ((*i).get_target() == e.m_target) {
            el.erase(i);
            return;
          }
      }

    } // namespace detail

    // O(1)
    template <class Config>
    inline std::pair<typename Config::edge_iterator,
                     typename Config::edge_iterator>
    edges(const directed_edges_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      typedef typename Config::edge_iterator edge_iterator;
      const graph_type& cg = static_cast<const graph_type&>(g_);
      graph_type& g = const_cast<graph_type&>(cg);
      return std::make_pair( edge_iterator(g.vertex_set().begin(),
                                           g.vertex_set().begin(),
                                           g.vertex_set().end(), g),
                             edge_iterator(g.vertex_set().begin(),
                                           g.vertex_set().end(),
                                           g.vertex_set().end(), g) );
    }

    // O(E/V)
    template <class Config>
    inline void
    remove_edge(typename Config::vertex_descriptor u,
                typename Config::vertex_descriptor v,
                directed_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      typedef typename Config::edge_parallel_category Cat;
      graph_type& g = static_cast<graph_type&>(g_);
      detail::erase_from_incidence_list(g.out_edge_list(u), v, Cat());
    }

    template <class Config, class Predicate>
    inline void
    remove_out_edge_if(typename Config::vertex_descriptor u, Predicate pred,
                       directed_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::out_edge_iterator first, last;
      boost::tie(first, last) = out_edges(u, g);
      typedef typename Config::edge_parallel_category edge_parallel_category;
      detail::remove_directed_edge_if_dispatch
        (first, last, g.out_edge_list(u), pred, edge_parallel_category());
    }

    template <class Config, class Predicate>
    inline void
    remove_edge_if(Predicate pred, directed_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);

      typename Config::vertex_iterator vi, vi_end;
      for (boost::tie(vi, vi_end) = vertices(g); vi != vi_end; ++vi)
        remove_out_edge_if(*vi, pred, g);
    }

    template <class EdgeOrIter, class Config>
    inline void
    remove_edge(EdgeOrIter e_or_iter, directed_graph_helper<Config>& g_)
    {
      g_.remove_edge(e_or_iter);
    }

    // O(V + E) for allow_parallel_edges
    // O(V * log(E/V)) for disallow_parallel_edges
    template <class Config>
    inline void
    clear_vertex(typename Config::vertex_descriptor u,
                 directed_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      typedef typename Config::edge_parallel_category Cat;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::vertex_iterator vi, viend;
      for (boost::tie(vi, viend) = vertices(g); vi != viend; ++vi)
        detail::erase_from_incidence_list(g.out_edge_list(*vi), u, Cat());
      g.out_edge_list(u).clear();
      // clear() should be a req of Sequence and AssociativeContainer,
      // or maybe just Container
    }

    template <class Config>
    inline void
    clear_out_edges(typename Config::vertex_descriptor u,
                    directed_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);
      g.out_edge_list(u).clear();
      // clear() should be a req of Sequence and AssociativeContainer,
      // or maybe just Container
    }

    // O(V), could do better...
    template <class Config>
    inline typename Config::edges_size_type
    num_edges(const directed_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      const graph_type& g = static_cast<const graph_type&>(g_);
      typename Config::edges_size_type num_e = 0;
      typename Config::vertex_iterator vi, vi_end;
      for (boost::tie(vi, vi_end) = vertices(g); vi != vi_end; ++vi)
        num_e += out_degree(*vi, g);
      return num_e;
    }
    // O(1) for allow_parallel_edge_tag
    // O(log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline std::pair<typename directed_graph_helper<Config>::edge_descriptor, bool>
    add_edge(typename Config::vertex_descriptor u,
             typename Config::vertex_descriptor v,
             const typename Config::edge_property_type& p,
             directed_graph_helper<Config>& g_)
    {
      typedef typename Config::edge_descriptor edge_descriptor;
      typedef typename Config::graph_type graph_type;
      typedef typename Config::StoredEdge StoredEdge;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::OutEdgeList::iterator i;
      bool inserted;
      boost::tie(i, inserted) = boost::graph_detail::push(g.out_edge_list(u),
                                            StoredEdge(v, p));
      return std::make_pair(edge_descriptor(u, v, &(*i).get_property()),
                            inserted);
    }
    // Did not use default argument here because that
    // causes Visual C++ to get confused.
    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool>
    add_edge(typename Config::vertex_descriptor u,
             typename Config::vertex_descriptor v,
             directed_graph_helper<Config>& g_)
    {
      typename Config::edge_property_type p;
      return add_edge(u, v, p, g_);
    }

    namespace detail {

      // O(E/V)
      template <class Graph, class EdgeList, class Vertex>
      inline void
      remove_edge_and_property(Graph& g, EdgeList& el, Vertex v,
                               boost::allow_parallel_edge_tag cat)
      {
        typedef typename Graph::global_edgelist_selector EdgeListS;
        BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

        typename EdgeList::iterator i = el.begin(), end = el.end();
        for (; i != end; ++i) {
          if ((*i).get_target() == v) {
            // NOTE: Wihtout this skip, this loop will double-delete properties
            // of loop edges. This solution is based on the observation that
            // the incidence edges of a vertex with a loop are adjacent in the
            // out edge list. This *may* actually hold for multisets also.
            bool skip = (boost::next(i) != end && i->get_iter() == boost::next(i)->get_iter());
            g.m_edges.erase((*i).get_iter());
            if (skip) ++i;
          }
        }
        detail::erase_from_incidence_list(el, v, cat);
      }

      // O(log(E/V))
      template <class Graph, class EdgeList, class Vertex>
      inline void
      remove_edge_and_property(Graph& g, EdgeList& el, Vertex v,
                               boost::disallow_parallel_edge_tag)
      {
        typedef typename Graph::global_edgelist_selector EdgeListS;
        BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

        typedef typename EdgeList::value_type StoredEdge;
        typename EdgeList::iterator i = el.find(StoredEdge(v)), end = el.end();
        if (i != end) {
          g.m_edges.erase((*i).get_iter());
          el.erase(i);
        }
      }

    } // namespace detail

    // Had to make these non-members to avoid accidental instantiation
    // on SGI MIPSpro C++
    template <class C>
    inline typename C::InEdgeList&
    in_edge_list(undirected_graph_helper<C>&,
                 typename C::vertex_descriptor v)
    {
      typename C::stored_vertex* sv = (typename C::stored_vertex*)v;
      return sv->m_out_edges;
    }
    template <class C>
    inline const typename C::InEdgeList&
    in_edge_list(const undirected_graph_helper<C>&,
                 typename C::vertex_descriptor v) {
      typename C::stored_vertex* sv = (typename C::stored_vertex*)v;
      return sv->m_out_edges;
    }

    // O(E/V)
    template <class EdgeOrIter, class Config>
    inline void
    remove_edge(EdgeOrIter e, undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      g_.remove_edge(e);
    }

    // O(E/V) or O(log(E/V))
    template <class Config>
    void
    remove_edge(typename Config::vertex_descriptor u,
                typename Config::vertex_descriptor v,
                undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);
      typedef typename Config::edge_parallel_category Cat;
      detail::remove_edge_and_property(g, g.out_edge_list(u), v, Cat());
      detail::erase_from_incidence_list(g.out_edge_list(v), u, Cat());
    }

    template <class Config, class Predicate>
    void
    remove_out_edge_if(typename Config::vertex_descriptor u, Predicate pred,
                       undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      typedef typename Config::OutEdgeList::value_type::property_type PropT;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::out_edge_iterator first, last;
      boost::tie(first, last) = out_edges(u, g);
      typedef typename Config::edge_parallel_category Cat;
      detail::undirected_remove_out_edge_if_dispatch<PropT>
        (g, first, last, g.out_edge_list(u), pred, Cat());
    }

    template <class Config, class Predicate>
    void
    remove_in_edge_if(typename Config::vertex_descriptor u, Predicate pred,
                      undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      remove_out_edge_if(u, pred, g_);
    }

    // O(E/V * E) or O(log(E/V) * E)
    template <class Predicate, class Config>
    void
    remove_edge_if(Predicate pred, undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::edge_iterator ei, ei_end, next;
      boost::tie(ei, ei_end) = edges(g);
      for (next = ei; ei != ei_end; ei = next) {
        ++next;
        if (pred(*ei))
          remove_edge(*ei, g);
      }
    }

    // O(1)
    template <class Config>
    inline std::pair<typename Config::edge_iterator,
                     typename Config::edge_iterator>
    edges(const undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      typedef typename Config::edge_iterator edge_iterator;
      const graph_type& cg = static_cast<const graph_type&>(g_);
      graph_type& g = const_cast<graph_type&>(cg);
      return std::make_pair( edge_iterator(g.m_edges.begin()),
                             edge_iterator(g.m_edges.end()) );
    }

    // O(1)
    template <class Config>
    inline typename Config::edges_size_type
    num_edges(const undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      const graph_type& g = static_cast<const graph_type&>(g_);
      return g.m_edges.size();
    }

    // O(E/V * E/V)
    template <class Config>
    inline void
    clear_vertex(typename Config::vertex_descriptor u,
                 undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);
      while (true) {
        typename Config::out_edge_iterator ei, ei_end;
        boost::tie(ei, ei_end) = out_edges(u, g);
        if (ei == ei_end) break;
        remove_edge(*ei, g);
      }
    }

    // O(1) for allow_parallel_edge_tag
    // O(log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool>
    add_edge(typename Config::vertex_descriptor u,
             typename Config::vertex_descriptor v,
             const typename Config::edge_property_type& p,
             undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::StoredEdge StoredEdge;
      typedef typename Config::edge_descriptor edge_descriptor;
      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);

      bool inserted;
      typename Config::EdgeContainer::value_type e(u, v, p);
      typename Config::EdgeContainer::iterator p_iter
        = graph_detail::push(g.m_edges, e).first;

      typename Config::OutEdgeList::iterator i;
      boost::tie(i, inserted) = boost::graph_detail::push(g.out_edge_list(u),
                                    StoredEdge(v, p_iter, &g.m_edges));
      if (inserted) {
        boost::graph_detail::push(g.out_edge_list(v), StoredEdge(u, p_iter, &g.m_edges));
        return std::make_pair(edge_descriptor(u, v, &p_iter->get_property()),
                              true);
      } else {
        g.m_edges.erase(p_iter);
        return std::make_pair
          (edge_descriptor(u, v, &i->get_iter()->get_property()), false);
      }
    }

    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool>
    add_edge(typename Config::vertex_descriptor u,
             typename Config::vertex_descriptor v,
             undirected_graph_helper<Config>& g_)
    {
      typename Config::edge_property_type p;
      return add_edge(u, v, p, g_);
    }

    // O(1)
    template <class Config>
    inline typename Config::degree_size_type
    degree(typename Config::vertex_descriptor u,
           const undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type Graph;
      const Graph& g = static_cast<const Graph&>(g_);
      return out_degree(u, g);
    }

    template <class Config>
    inline std::pair<typename Config::in_edge_iterator,
                     typename Config::in_edge_iterator>
    in_edges(typename Config::vertex_descriptor u,
             const undirected_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type Graph;
      const Graph& cg = static_cast<const Graph&>(g_);
      Graph& g = const_cast<Graph&>(cg);
      typedef typename Config::in_edge_iterator in_edge_iterator;
      return
        std::make_pair(in_edge_iterator(g.out_edge_list(u).begin(), u),
                       in_edge_iterator(g.out_edge_list(u).end(), u));
    }

    template <class Config>
    inline typename Config::degree_size_type
    in_degree(typename Config::vertex_descriptor u,
              const undirected_graph_helper<Config>& g_)
    { return degree(u, g_); }

    template <class C>
    inline typename C::InEdgeList&
    in_edge_list(bidirectional_graph_helper<C>&,
                 typename C::vertex_descriptor v)
    {
      typename C::stored_vertex* sv = (typename C::stored_vertex*)v;
      return sv->m_in_edges;
    }

    template <class C>
    inline const typename C::InEdgeList&
    in_edge_list(const bidirectional_graph_helper<C>&,
                 typename C::vertex_descriptor v) {
      typename C::stored_vertex* sv = (typename C::stored_vertex*)v;
      return sv->m_in_edges;
    }

    template <class Predicate, class Config>
    inline void
    remove_edge_if(Predicate pred, bidirectional_graph_helper<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::edge_iterator ei, ei_end, next;
      boost::tie(ei, ei_end) = edges(g);
      for (next = ei; ei != ei_end; ei = next) {
        ++next;
        if (pred(*ei))
          remove_edge(*ei, g);
      }
    }

    template <class Config>
    inline std::pair<typename Config::in_edge_iterator,
                     typename Config::in_edge_iterator>
    in_edges(typename Config::vertex_descriptor u,
             const bidirectional_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      const graph_type& cg = static_cast<const graph_type&>(g_);
      graph_type& g = const_cast<graph_type&>(cg);
      typedef typename Config::in_edge_iterator in_edge_iterator;
      return
        std::make_pair(in_edge_iterator(in_edge_list(g, u).begin(), u),
                       in_edge_iterator(in_edge_list(g, u).end(), u));
    }

    // O(1)
    template <class Config>
    inline std::pair<typename Config::edge_iterator,
                     typename Config::edge_iterator>
    edges(const bidirectional_graph_helper<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      typedef typename Config::edge_iterator edge_iterator;
      const graph_type& cg = static_cast<const graph_type&>(g_);
      graph_type& g = const_cast<graph_type&>(cg);
      return std::make_pair( edge_iterator(g.m_edges.begin()),
                             edge_iterator(g.m_edges.end()) );
    }

    // O(E/V) for allow_parallel_edge_tag
    // O(log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline void
    remove_edge(typename Config::vertex_descriptor u,
                typename Config::vertex_descriptor v,
                bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);
      typedef typename Config::edge_parallel_category Cat;
      detail::remove_edge_and_property(g, g.out_edge_list(u), v, Cat());
      detail::erase_from_incidence_list(in_edge_list(g, v), u, Cat());
    }

    // O(E/V) or O(log(E/V))
    template <class EdgeOrIter, class Config>
    inline void
    remove_edge(EdgeOrIter e,
                bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      g_.remove_edge(e);
    }

    template <class Config, class Predicate>
    inline void
    remove_out_edge_if(typename Config::vertex_descriptor u, Predicate pred,
                       bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      typedef typename Config::OutEdgeList::value_type::property_type PropT;
      graph_type& g = static_cast<graph_type&>(g_);

      typedef typename Config::EdgeIter EdgeIter;
      typedef std::vector<EdgeIter> Garbage;
      Garbage garbage;

      // First remove the edges from the targets' in-edge lists and
      // from the graph's edge set list.
      typename Config::out_edge_iterator out_i, out_end;
      for (boost::tie(out_i, out_end) = out_edges(u, g); out_i != out_end; ++out_i)
        if (pred(*out_i)) {
          detail::remove_directed_edge_dispatch
            (*out_i, in_edge_list(g, target(*out_i, g)),
             *(PropT*)(*out_i).get_property());
          // Put in garbage to delete later. Will need the properties
          // for the remove_if of the out-edges.
          garbage.push_back((*out_i.base()).get_iter());
        }

      // Now remove the edges from this out-edge list.
      typename Config::out_edge_iterator first, last;
      boost::tie(first, last) = out_edges(u, g);
      typedef typename Config::edge_parallel_category Cat;
      detail::remove_directed_edge_if_dispatch
        (first, last, g.out_edge_list(u), pred, Cat());

      // Now delete the edge properties from the g.m_edges list
      for (typename Garbage::iterator i = garbage.begin();
           i != garbage.end(); ++i)
        g.m_edges.erase(*i);
    }

    template <class Config, class Predicate>
    inline void
    remove_in_edge_if(typename Config::vertex_descriptor v, Predicate pred,
                      bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      typedef typename Config::OutEdgeList::value_type::property_type PropT;
      graph_type& g = static_cast<graph_type&>(g_);

      typedef typename Config::EdgeIter EdgeIter;
      typedef std::vector<EdgeIter> Garbage;
      Garbage garbage;

      // First remove the edges from the sources' out-edge lists and
      // from the graph's edge set list.
      typename Config::in_edge_iterator in_i, in_end;
      for (boost::tie(in_i, in_end) = in_edges(v, g); in_i != in_end; ++in_i)
        if (pred(*in_i)) {
          typename Config::vertex_descriptor u = source(*in_i, g);
          detail::remove_directed_edge_dispatch
            (*in_i, g.out_edge_list(u), *(PropT*)(*in_i).get_property());
          // Put in garbage to delete later. Will need the properties
          // for the remove_if of the out-edges.
          garbage.push_back((*in_i.base()).get_iter());
        }
      // Now remove the edges from this in-edge list.
      typename Config::in_edge_iterator first, last;
      boost::tie(first, last) = in_edges(v, g);
      typedef typename Config::edge_parallel_category Cat;
      detail::remove_directed_edge_if_dispatch
        (first, last, in_edge_list(g, v), pred, Cat());

      // Now delete the edge properties from the g.m_edges list
      for (typename Garbage::iterator i = garbage.begin();
           i != garbage.end(); ++i)
        g.m_edges.erase(*i);
    }

    // O(1)
    template <class Config>
    inline typename Config::edges_size_type
    num_edges(const bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      const graph_type& g = static_cast<const graph_type&>(g_);
      return g.m_edges.size();
    }

    // O(E/V * E/V) for allow_parallel_edge_tag
    // O(E/V * log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline void
    clear_vertex(typename Config::vertex_descriptor u,
                 bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      typedef typename Config::edge_parallel_category Cat;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::OutEdgeList& el = g.out_edge_list(u);
      typename Config::OutEdgeList::iterator
        ei = el.begin(), ei_end = el.end();
      for (; ei != ei_end; ++ei) {
        detail::erase_from_incidence_list
          (in_edge_list(g, (*ei).get_target()), u, Cat());
        g.m_edges.erase((*ei).get_iter());
      }
      typename Config::InEdgeList& in_el = in_edge_list(g, u);
      typename Config::InEdgeList::iterator
        in_ei = in_el.begin(), in_ei_end = in_el.end();
      for (; in_ei != in_ei_end; ++in_ei) {
        detail::erase_from_incidence_list
          (g.out_edge_list((*in_ei).get_target()), u, Cat());
        g.m_edges.erase((*in_ei).get_iter());
      }
      g.out_edge_list(u).clear();
      in_edge_list(g, u).clear();
    }

    template <class Config>
    inline void
    clear_out_edges(typename Config::vertex_descriptor u,
                    bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      typedef typename Config::edge_parallel_category Cat;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::OutEdgeList& el = g.out_edge_list(u);
      typename Config::OutEdgeList::iterator
        ei = el.begin(), ei_end = el.end();
      for (; ei != ei_end; ++ei) {
        detail::erase_from_incidence_list
          (in_edge_list(g, (*ei).get_target()), u, Cat());
        g.m_edges.erase((*ei).get_iter());
      }
      g.out_edge_list(u).clear();
    }

    template <class Config>
    inline void
    clear_in_edges(typename Config::vertex_descriptor u,
                   bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::global_edgelist_selector EdgeListS;
      BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

      typedef typename Config::graph_type graph_type;
      typedef typename Config::edge_parallel_category Cat;
      graph_type& g = static_cast<graph_type&>(g_);
      typename Config::InEdgeList& in_el = in_edge_list(g, u);
      typename Config::InEdgeList::iterator
        in_ei = in_el.begin(), in_ei_end = in_el.end();
      for (; in_ei != in_ei_end; ++in_ei) {
        detail::erase_from_incidence_list
          (g.out_edge_list((*in_ei).get_target()), u, Cat());
        g.m_edges.erase((*in_ei).get_iter());
      }
      in_edge_list(g, u).clear();
    }

    // O(1) for allow_parallel_edge_tag
    // O(log(E/V)) for disallow_parallel_edge_tag
    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool>
    add_edge(typename Config::vertex_descriptor u,
             typename Config::vertex_descriptor v,
             const typename Config::edge_property_type& p,
             bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      graph_type& g = static_cast<graph_type&>(g_);
      typedef typename Config::edge_descriptor edge_descriptor;
      typedef typename Config::StoredEdge StoredEdge;
      bool inserted;
      typename Config::EdgeContainer::value_type e(u, v, p);
      typename Config::EdgeContainer::iterator p_iter
        = graph_detail::push(g.m_edges, e).first;
      typename Config::OutEdgeList::iterator i;
      boost::tie(i, inserted) = boost::graph_detail::push(g.out_edge_list(u),
                                        StoredEdge(v, p_iter, &g.m_edges));
      if (inserted) {
        boost::graph_detail::push(in_edge_list(g, v), StoredEdge(u, p_iter, &g.m_edges));
        return std::make_pair(edge_descriptor(u, v, &p_iter->m_property),
                              true);
      } else {
        g.m_edges.erase(p_iter);
        return std::make_pair(edge_descriptor(u, v,
                                     &i->get_iter()->get_property()),
                              false);
      }
    }

    template <class Config>
    inline std::pair<typename Config::edge_descriptor, bool>
    add_edge(typename Config::vertex_descriptor u,
             typename Config::vertex_descriptor v,
             bidirectional_graph_helper_with_property<Config>& g_)
    {
      typename Config::edge_property_type p;
      return add_edge(u, v, p, g_);
    }
    // O(1)
    template <class Config>
    inline typename Config::degree_size_type
    degree(typename Config::vertex_descriptor u,
           const bidirectional_graph_helper_with_property<Config>& g_)
    {
      typedef typename Config::graph_type graph_type;
      const graph_type& g = static_cast<const graph_type&>(g_);
      return in_degree(u, g) + out_degree(u, g);
    }

    template <class Config, class Base>
    inline std::pair<typename Config::adjacency_iterator,
                     typename Config::adjacency_iterator>
    adjacent_vertices(typename Config::vertex_descriptor u,
                      const adj_list_helper<Config, Base>& g_)
    {
      typedef typename Config::graph_type AdjList;
      const AdjList& cg = static_cast<const AdjList&>(g_);
      AdjList& g = const_cast<AdjList&>(cg);
      typedef typename Config::adjacency_iterator adjacency_iterator;
      typename Config::out_edge_iterator first, last;
      boost::tie(first, last) = out_edges(u, g);
      return std::make_pair(adjacency_iterator(first, &g),
                            adjacency_iterator(last, &g));
    }

    template <class Config, class Base>
    inline std::pair<typename Config::inv_adjacency_iterator,
                     typename Config::inv_adjacency_iterator>
    inv_adjacent_vertices(typename Config::vertex_descriptor u,
                          const adj_list_helper<Config, Base>& g_)
    {
      typedef typename Config::graph_type AdjList;
      const AdjList& cg = static_cast<const AdjList&>(g_);
      AdjList& g = const_cast<AdjList&>(cg);
      typedef typename Config::inv_adjacency_iterator inv_adjacency_iterator;
      typename Config::in_edge_iterator first, last;
      boost::tie(first, last) = in_edges(u, g);
      return std::make_pair(inv_adjacency_iterator(first, &g),
                            inv_adjacency_iterator(last, &g));
    }

    template <class Config, class Base>
    inline std::pair<typename Config::out_edge_iterator,
                     typename Config::out_edge_iterator>
    out_edges(typename Config::vertex_descriptor u,
              const adj_list_helper<Config, Base>& g_)
    {
      typedef typename Config::graph_type AdjList;
      typedef typename Config::out_edge_iterator out_edge_iterator;
      const AdjList& cg = static_cast<const AdjList&>(g_);
      AdjList& g = const_cast<AdjList&>(cg);
      return
        std::make_pair(out_edge_iterator(g.out_edge_list(u).begin(), u),
                       out_edge_iterator(g.out_edge_list(u).end(), u));
    }

    template <class Config, class Base>
    inline std::pair<typename Config::vertex_iterator,
                     typename Config::vertex_iterator>
    vertices(const adj_list_helper<Config, Base>& g_)
    {
      typedef typename Config::graph_type AdjList;
      const AdjList& cg = static_cast<const AdjList&>(g_);
      AdjList& g = const_cast<AdjList&>(cg);
      return std::make_pair( g.vertex_set().begin(), g.vertex_set().end() );
    }

    template <class Config, class Base>
    inline typename Config::vertices_size_type
    num_vertices(const adj_list_helper<Config, Base>& g_)
    {
      typedef typename Config::graph_type AdjList;
      const AdjList& g = static_cast<const AdjList&>(g_);
      return g.vertex_set().size();
    }

    template <class Config, class Base>
    inline typename Config::degree_size_type
    out_degree(typename Config::vertex_descriptor u,
               const adj_list_helper<Config, Base>& g_)
    {
      typedef typename Config::graph_type AdjList;
      const AdjList& g = static_cast<const AdjList&>(g_);
      return g.out_edge_list(u).size();
    }

    template <class Config, class Base>
    inline std::pair<typename Config::edge_descriptor, bool>
    edge(typename Config::vertex_descriptor u,
         typename Config::vertex_descriptor v,
         const adj_list_helper<Config, Base>& g_)
    {
      typedef typename Config::graph_type Graph;
      typedef typename Config::StoredEdge StoredEdge;
      const Graph& cg = static_cast<const Graph&>(g_);
      const typename Config::OutEdgeList& el = cg.out_edge_list(u);
      typename Config::OutEdgeList::const_iterator it = graph_detail::
        find(el, StoredEdge(v));
      return std::make_pair(
               typename Config::edge_descriptor
                     (u, v, (it == el.end() ? 0 : &(*it).get_property())),
               (it != el.end()));
    }

    template <class Config, class Base>
    inline std::pair<typename Config::out_edge_iterator,
                     typename Config::out_edge_iterator>
    edge_range(typename Config::vertex_descriptor u,
               typename Config::vertex_descriptor v,
               const adj_list_helper<Config, Base>& g_)
    {
      typedef typename Config::graph_type Graph;
      typedef typename Config::StoredEdge StoredEdge;
      const Graph& cg = static_cast<const Graph&>(g_);
      Graph& g = const_cast<Graph&>(cg);
      typedef typename Config::out_edge_iterator out_edge_iterator;
      typename Config::OutEdgeList& el = g.out_edge_list(u);
      typename Config::OutEdgeList::iterator first, last;
      typename Config::EdgeContainer fake_edge_container;
      boost::tie(first, last) = graph_detail::
        equal_range(el, StoredEdge(v));
      return std::make_pair(out_edge_iterator(first, u),
                            out_edge_iterator(last, u));
    }

    template <class Config>
    inline typename Config::degree_size_type
    in_degree(typename Config::vertex_descriptor u,
              const directed_edges_helper<Config>& g_)
    {
      typedef typename Config::graph_type Graph;
      const Graph& cg = static_cast<const Graph&>(g_);
      Graph& g = const_cast<Graph&>(cg);
      return in_edge_list(g, u).size();
    }

    namespace detail {
      template <class Config, class Base, class Property>
      inline
      typename boost::property_map<typename Config::graph_type,
        Property>::type
      get_dispatch(adj_list_helper<Config,Base>&, Property p,
                   boost::edge_property_tag) {
        typedef typename Config::graph_type Graph;
        typedef typename boost::property_map<Graph, Property>::type PA;
        return PA(p);
      }

      template <class Config, class Base, class Property>
      inline
      typename boost::property_map<typename Config::graph_type,
        Property>::const_type
      get_dispatch(const adj_list_helper<Config,Base>&, Property p,
                   boost::edge_property_tag) {
        typedef typename Config::graph_type Graph;
        typedef typename boost::property_map<Graph, Property>::const_type PA;
        return PA(p);
      }

      template <class Config, class Base, class Property>
      inline
      typename boost::property_map<typename Config::graph_type,
        Property>::type
      get_dispatch(adj_list_helper<Config,Base>& g, Property p,
                   boost::vertex_property_tag) {
        typedef typename Config::graph_type Graph;
        typedef typename boost::property_map<Graph, Property>::type PA;
        return PA(&static_cast<Graph&>(g), p);
      }

      template <class Config, class Base, class Property>
      inline
      typename boost::property_map<typename Config::graph_type,
        Property>::const_type
      get_dispatch(const adj_list_helper<Config, Base>& g, Property p,
                   boost::vertex_property_tag) {
        typedef typename Config::graph_type Graph;
        typedef typename boost::property_map<Graph, Property>::const_type PA;
        const Graph& cg = static_cast<const Graph&>(g);
        return PA(&cg, p);
      }

    } // namespace detail

    // Implementation of the PropertyGraph interface
    template <class Config, class Base, class Property>
    inline
    typename boost::property_map<typename Config::graph_type, Property>::type
    get(Property p, adj_list_helper<Config, Base>& g) {
      typedef typename detail::property_kind_from_graph<adj_list_helper<Config, Base>, Property>::type Kind;
      return detail::get_dispatch(g, p, Kind());
    }

    template <class Config, class Base, class Property>
    inline
    typename boost::property_map<typename Config::graph_type,
      Property>::const_type
    get(Property p, const adj_list_helper<Config, Base>& g) {
      typedef typename detail::property_kind_from_graph<adj_list_helper<Config, Base>, Property>::type Kind;
      return detail::get_dispatch(g, p, Kind());
    }

    template <class Config, class Base, class Property, class Key>
    inline
    typename boost::property_traits<
      typename boost::property_map<typename Config::graph_type,
        Property>::type
    >::reference
    get(Property p, adj_list_helper<Config, Base>& g, const Key& key) {
      return get(get(p, g), key);
    }

    template <class Config, class Base, class Property, class Key>
    inline
    typename boost::property_traits<
      typename boost::property_map<typename Config::graph_type,
        Property>::const_type
    >::reference
    get(Property p, const adj_list_helper<Config, Base>& g, const Key& key) {
      return get(get(p, g), key);
    }

    template <class Config, class Base, class Property, class Key,class Value>
    inline void
    put(Property p, adj_list_helper<Config, Base>& g,
        const Key& key, const Value& value)
    {
      typedef typename Config::graph_type Graph;
      typedef typename boost::property_map<Graph, Property>::type Map;
      Map pmap = get(p, static_cast<Graph&>(g));
      put(pmap, key, value);
    }


    // O(1)
    template <class Derived, class Config, class Base>
    inline typename Config::vertex_descriptor
    add_vertex(adj_list_impl<Derived, Config, Base>& g_)
    {
      Derived& g = static_cast<Derived&>(g_);
      typedef typename Config::stored_vertex stored_vertex;
      stored_vertex* v = new stored_vertex;
      typename Config::StoredVertexList::iterator pos;
      bool inserted;
      boost::tie(pos,inserted) = boost::graph_detail::push(g.m_vertices, v);
      v->m_position = pos;
      g.added_vertex(v);
      return v;
    }

    // O(1)
    template <class Derived, class Config, class Base>
    inline typename Config::vertex_descriptor
    add_vertex(const typename Config::vertex_property_type& p,
               adj_list_impl<Derived, Config, Base>& g_)
    {
      typedef typename Config::vertex_descriptor vertex_descriptor;
      Derived& g = static_cast<Derived&>(g_);
      if (optional<vertex_descriptor> v
            = g.vertex_by_property(get_property_value(p, vertex_bundle)))
        return *v;

      typedef typename Config::stored_vertex stored_vertex;
      stored_vertex* v = new stored_vertex(p);
      typename Config::StoredVertexList::iterator pos;
      bool inserted;
      boost::tie(pos,inserted) = boost::graph_detail::push(g.m_vertices, v);
      v->m_position = pos;
      g.added_vertex(v);
      return v;
    }

    // O(1)
    template <class Derived, class Config, class Base>
    inline void remove_vertex(typename Config::vertex_descriptor u,
                              adj_list_impl<Derived, Config, Base>& g_)
    {
      typedef typename Config::stored_vertex stored_vertex;
      Derived& g = static_cast<Derived&>(g_);
      g.removing_vertex(u, boost::graph_detail::iterator_stability(g_.m_vertices));
      stored_vertex* su = (stored_vertex*)u;
      g.m_vertices.erase(su->m_position);
      delete su;
    }

    // O(V)
    template <class Derived, class Config, class Base>
    inline typename Config::vertex_descriptor
    vertex(typename Config::vertices_size_type n,
           const adj_list_impl<Derived, Config, Base>& g_)
    {
      const Derived& g = static_cast<const Derived&>(g_);
      typename Config::vertex_iterator i = vertices(g).first;
      while (n--) ++i; // std::advance(i, n); (not VC++ portable)
      return *i;
    }

    namespace detail {

      template <class Graph, class vertex_descriptor>
      inline void
      remove_vertex_dispatch(Graph& g, vertex_descriptor u,
                             boost::directed_tag)
      {
        typedef typename Graph::edge_parallel_category edge_parallel_category;
        g.m_vertices.erase(g.m_vertices.begin() + u);
        vertex_descriptor V = num_vertices(g);
        if (u != V) {
          for (vertex_descriptor v = 0; v < V; ++v)
            reindex_edge_list(g.out_edge_list(v), u, edge_parallel_category());
        }
      }

      template <class Graph, class vertex_descriptor>
      inline void
      remove_vertex_dispatch(Graph& g, vertex_descriptor u,
                             boost::undirected_tag)
      {
        typedef typename Graph::global_edgelist_selector EdgeListS;
        BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

        typedef typename Graph::edge_parallel_category edge_parallel_category;
        g.m_vertices.erase(g.m_vertices.begin() + u);
        vertex_descriptor V = num_vertices(g);
        for (vertex_descriptor v = 0; v < V; ++v)
          reindex_edge_list(g.out_edge_list(v), u,
                            edge_parallel_category());
        typedef typename Graph::EdgeContainer Container;
        typedef typename Container::iterator Iter;
        Iter ei = g.m_edges.begin(), ei_end = g.m_edges.end();
        for (; ei != ei_end; ++ei) {
          if (ei->m_source > u)
            --ei->m_source;
          if (ei->m_target > u)
            --ei->m_target;
        }
      }

      template <class Graph, class vertex_descriptor>
      inline void
      remove_vertex_dispatch(Graph& g, vertex_descriptor u,
                             boost::bidirectional_tag)
      {
        typedef typename Graph::global_edgelist_selector EdgeListS;
        BOOST_STATIC_ASSERT((!is_same<EdgeListS, vecS>::value));

        typedef typename Graph::edge_parallel_category edge_parallel_category;
        g.m_vertices.erase(g.m_vertices.begin() + u);
        vertex_descriptor V = num_vertices(g);
        vertex_descriptor v;
        if (u != V) {
          for (v = 0; v < V; ++v)
            reindex_edge_list(g.out_edge_list(v), u,
                              edge_parallel_category());
          for (v = 0; v < V; ++v)
            reindex_edge_list(in_edge_list(g, v), u,
                              edge_parallel_category());

          typedef typename Graph::EdgeContainer Container;
          typedef typename Container::iterator Iter;
          Iter ei = g.m_edges.begin(), ei_end = g.m_edges.end();
          for (; ei != ei_end; ++ei) {
            if (ei->m_source > u)
              --ei->m_source;
            if (ei->m_target > u)
              --ei->m_target;
          }
        }
      }

      template <class EdgeList, class vertex_descriptor>
      inline void
      reindex_edge_list(EdgeList& el, vertex_descriptor u,
                        boost::allow_parallel_edge_tag)
      {
        typename EdgeList::iterator ei = el.begin(), e_end = el.end();
        for (; ei != e_end; ++ei)
          if ((*ei).get_target() > u)
            --(*ei).get_target();
      }

      template <class EdgeList, class vertex_descriptor>
      inline void
      reindex_edge_list(EdgeList& el, vertex_descriptor u,
                        boost::disallow_parallel_edge_tag)
      {
        typename EdgeList::iterator ei = el.begin(), e_end = el.end();
        while (ei != e_end) {
          typename EdgeList::value_type ce = *ei;
          ++ei;
          if (ce.get_target() > u) {
            el.erase(ce);
            --ce.get_target();
            el.insert(ce);
          }
        }
      }
    } // namespace detail

    template <class G, class C, class B>
    inline typename C::InEdgeList&
    in_edge_list(vec_adj_list_impl<G,C,B>& g,
                 typename C::vertex_descriptor v) {
      return g.m_vertices[v].m_in_edges;
    }
    template <class G, class C, class B>
    inline const typename C::InEdgeList&
    in_edge_list(const vec_adj_list_impl<G,C,B>& g,
                 typename C::vertex_descriptor v) {
      return g.m_vertices[v].m_in_edges;
    }

    // O(1)
    template <class Graph, class Config, class Base>
    inline typename Config::vertex_descriptor
    add_vertex(vec_adj_list_impl<Graph, Config, Base>& g_) {
      Graph& g = static_cast<Graph&>(g_);
      g.m_vertices.resize(g.m_vertices.size() + 1);
      g.added_vertex(g.m_vertices.size() - 1);
      return g.m_vertices.size() - 1;
    }

    template <class Graph, class Config, class Base>
    inline typename Config::vertex_descriptor
    add_vertex(const typename Config::vertex_property_type& p,
               vec_adj_list_impl<Graph, Config, Base>& g_) {
      typedef typename Config::vertex_descriptor vertex_descriptor;
      Graph& g = static_cast<Graph&>(g_);
      if (optional<vertex_descriptor> v
            = g.vertex_by_property(get_property_value(p, vertex_bundle)))
        return *v;
      typedef typename Config::stored_vertex stored_vertex;
      g.m_vertices.push_back(stored_vertex(p));
      g.added_vertex(g.m_vertices.size() - 1);
      return g.m_vertices.size() - 1;
    }

    template <class Graph, class Config, class Base>
    inline std::pair<typename Config::edge_descriptor, bool>
    add_edge(typename Config::vertex_descriptor u,
             typename Config::vertex_descriptor v,
             const typename Config::edge_property_type& p,
             vec_adj_list_impl<Graph, Config, Base>& g_)
    {
      BOOST_USING_STD_MAX();
      typename Config::vertex_descriptor x = max BOOST_PREVENT_MACRO_SUBSTITUTION(u, v);
      if (x >= num_vertices(g_))
        g_.m_vertices.resize(x + 1);
      adj_list_helper<Config, Base>& g = g_;
      return add_edge(u, v, p, g);
    }

    template <class Graph, class Config, class Base>
    inline std::pair<typename Config::edge_descriptor, bool>
    add_edge(typename Config::vertex_descriptor u,
             typename Config::vertex_descriptor v,
             vec_adj_list_impl<Graph, Config, Base>& g_)
    {
      typename Config::edge_property_type p;
      return add_edge(u, v, p, g_);
    }

    // O(V + E)
    template <class Graph, class Config, class Base>
    inline void remove_vertex(typename Config::vertex_descriptor v,
                              vec_adj_list_impl<Graph, Config, Base>& g_)
    {
      typedef typename Config::directed_category Cat;
      Graph& g = static_cast<Graph&>(g_);
      g.removing_vertex(v, boost::graph_detail::iterator_stability(g_.m_vertices));
      detail::remove_vertex_dispatch(g, v, Cat());
    }

    // O(1)
    template <class Graph, class Config, class Base>
    inline typename Config::vertex_descriptor
    vertex(typename Config::vertices_size_type n,
           const vec_adj_list_impl<Graph, Config, Base>&)
    {
      return n;
    }

} // namespace boost
