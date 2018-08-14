#include <boost/graph/adjacency_list.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

template <class T>
struct edge_t {
  edge_t(T& g) : G(g) {}

  typedef typename boost::graph_traits<T>::edge_descriptor Edge;
  typedef typename boost::graph_traits<T>::vertex_descriptor Vertex;

  void operator()(Edge e) const {
    Vertex source = boost::source(e, G);
    Vertex target = boost::target(e, G);

    std::cout << "From: " << source << ", To: " << target << "\n";
  }

  T& G;
};

auto main() -> decltype(0) {
  typedef boost::adjacency_list<> G;
  std::vector<std::pair<int, int> > edges;
  G g;

  for (int i = 1; i <= 3; ++i) {
    for (int j = 3; j >= 1; --j) {
      std::pair<int, int> edge(i, j);
      edges.push_back(edge);
    }
  }

  auto it = edges.begin();

  for (; it < edges.end(); ++it) {
    boost::add_edge(it->first, it->second, g);
  }

  std::for_each(boost::edges(g).first, boost::edges(g).second, edge_t<G>(g));

  return 0;
}

