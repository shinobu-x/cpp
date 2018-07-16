#include <algorithm>
#include <iostream>
#include <vector>
/**
 * 全域木 - Spanning Tree
 *
 * グラフ G(V, E) において、T ⊆ E となる辺集合 T があるとき、グラフ S(V, T) が木
 * （閉路を持たないグラフ）であるなら、S(V, T) のことをグラフ G(V, E) の全域木で
 * あるという。
 *
 * つまり、あるグラフの全ての頂点と、そのグラフを構成する辺の一部のみで構成され
 * る木のこと。
 *
 * 最小全域木
 *
 * 各辺に重みがある場合、最小の総和コストで構成される全域木。
 *                                                               引用：Wikipedia
 */
/**
 * UnionFindアルゴリズム
 *
 * データの集合を素集合（互いにオーバーラップしない集合）に分割して保存するデー
 * タ構造に対して、以下の機能を提供する。
 *
 *  - Find: 特定の要素がどの集合に属しているかを求める。
 *  - Union: 2つの集合を1つに統合する。
 *                                                               引用：Wikipedia
 */
template <typename T>
struct UnionFind {
  std::vector<T> par;
  std::vector<T> rank;

  // ノードの初期化
  UnionFind(T n) : par(n), rank(n, 1) {
    for (T i = 0; i < n; ++i) {
      par[i] = i;
    }
  }

  // データが属する木の根を探索
  T Find(T x) {
    if (par[x] == x) {
      return x;
    }
    return par[x] = Find(par[x]);
  }

  // 2つのデータが属する木を統合
  void Union(T x, T y) {
    x = Find(x);
    y = Find(y);

    // 同じ根に属する場合
    if (x == y) {
      return;
    }

    // 木のサイズが大きい方に統合
    if (rank[x] < rank[y]) {
      par[x] = y;
    } else {
      par[y] = x;
      // 木のサイズが同じ場合
      if (rank[x] == rank[y]) {
        rank[x]++;
      }
    }
  }

  // ノード間比較
  bool Same(T x, T y) {
    return Find(x) == Find(y);
  }
};

// グラフに含まれる2頂点と2点間接続コストを保持する構造体
template <typename T>
struct Edge {
  T e1;
  T e2;
  T cost;
  // 昇順ソートの際に使用
  bool operator<(const Edge e) {
    return cost < e.cost;
  }
};

// グラフを構成する構造体
template <typename T>
struct Graph {
  T n;
  std::vector<Edge<int> > e;
};

/**
 * クラスカル法
 *
 * グラフの各頂点が、それぞれの木に属するように森（木の集合）F を生成する。
 * つまり、頂点1個だけからなる木が頂点の個数だけ存在するということ。
 *                                                               引用：Wikipedia
 */
template <typename T>
T Kruskal(Graph<T>* g) {
  // コストが小さい順にソート
  std::sort(g->e.begin(), g->e.end());

  // ノードの作成
  UnionFind<T> uf(g->n);
  // コストの初期化
  T cost = 0;

  // 頂点同士を比較
  for (int i = 0; i < g->e.size(); ++i) {
    Edge<int>& e = g->e[i];
    // 閉路にならない場合
    if (!uf.Same(e.e1, e.e2)) {
      //std::cout << "e1: " << e.e1 << " - " << "e2: " << e.e2
      //          << " / cost: " << e.cost << "\n";
      uf.Union(e.e1, e.e2);
      cost += e.cost;
    }
  }

  return cost;
}

auto main() -> decltype(0) {
  Graph<int> g;

  // 超点数
  std::cin >> g.n;
  // 頂点の生成
  /**
   * 例
   *  頂点：0, 1, 2, 3, ,4
   *  コスト:
   *   0 - 1: 7
   *   0 - 3: 3
   *   1 - 2: 4
   *   1 - 3: 50
   *   1 - 4: 100
   *   2 - 3: 500
   *   2 - 4: 1000
   *   3 - 4: 2000
   *
   * 入力例
   *   8
   *   0 1 7
   *   0 3 3
   *   1 2 4
   *   1 3 50
   *   1 4 100
   *   2 3 500
   *   2 4 1000
   *   3 4 2000
   */
  for (int i = 0; i < g.n; ++i) {
    Edge<int> e;
    std::cin >> e.e1 >> e.e2 >> e.cost;
    g.e.push_back(e);
  }

  auto c = Kruskal<int>(&g);
  std::cout << c << "\n";

  return 0;
}
