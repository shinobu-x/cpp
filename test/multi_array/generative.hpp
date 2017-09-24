#include <boost/multi_array.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {
  unsigned int test = 0;
}

struct mutable_array_tag {};
struct const_array_tag {};

template <typename array_t>
void assign_if_not_const(array_t&, const const_array_tag&) {}

template <typename array_t>
void assign_if_not_const(array_t& a, const mutable_array_tag&);

template <typename array_t>
void assign_if_not_const(array_t& a, const mutable_array_tag&) {
  typedef typename array_t::index index;
  const index idx0 = a.index_bases()[0];
  const index idx1 = a.index_bases()[1];
  const index idx2 = a.index_bases()[2];

  int num = 0;
  for (index i = idx0; i != idx0 + 2; ++i)
    for (index j = idx1; j != idx1 + 3; ++j)
      for (index k = idx2; k != idx2 + 4; ++k)
        a[i][j][k] = ++num;
}

template <typename array_t>
void assign(array_t& a) {
  assign_if_not_const(a, mutable_array_tag());
}

template <typename array_t>
void access(array_t& a, const mutable_array_tag&);

template <typename array_t>
void access(array_t& a, const const_array_tag&);

template <typename storage_order3, typename storage_order4, typename modifier_t>
int do_configuration(const storage_order3& so3, const storage_order4& so4,
  const modifier_t& modifier) {
  {
    typedef boost::multi_array<int, 3> array;
    typename array::extent_gen extents;
    {
      array a(extents[2][3][4], so3);
      modifier.modify(a);
      access(a, mutable_array_tag());
    }
  }
  {
    typedef boost::multi_array_ref<int, 3> array_ref;
    typename array_ref::extent_gen extents;
    {
      int local[24];
      array_ref a(local, extents[2][3][4], so3);
      modifier.modify(a);
      assign(a);
      access(a, mutable_array_tag());
    }
  }
  {
    typedef boost::multi_array_ref<int, 3> array_ref;
    typedef boost::const_multi_array_ref<int, 3> const_array_ref;
    typename array_ref::extent_gen extents;
    {
      int local[24];
      array_ref a(local, extents[2][3][4], so3);
      modifier.modify(a);
      assign(a);
      const_array_ref b = a;
      access(a, const_array_tag());
    }
  }
  {
    typedef boost::multi_array<int, 4> array;
    typename array::extent_gen extents;
    {
      array a(extents[2][2][3][4], so4);
      modifier.modify(a);
      typename array::template subarray<3>::type b = a[1];
      assign(b);
      typename array::template const_subarray<3>::type c = b;
      access(c, const_array_tag());
    }
  }
  {
    typedef boost::multi_array<int, 3> array;
    typedef typename array::index_range range;
    typename array::index_gen indices;
    typename array::extent_gen extents;
    {
      typedef typename array::index index;
      array a(extents[4][5][6], so3);
      modifier.modify(a);
      const index idx0 = a.index_bases()[0];
      const index idx1 = a.index_bases()[1];
      const index idx2 = a.index_bases()[2];

      typename array::template array_view<3>::type b = a[
        indices[range(idx0 + 1, idx0 + 3)]
               [range(idx1 + 1, idx1 + 4)]
               [range(idx2 + 2, idx2 + 5)]
      ];
      access(b, mutable_array_tag());
    }
  }
  {
    typedef boost::multi_array<int, 3> array;
    typedef typename array::index_range range;
    typename array::index_gen indices;
    typename array::extent_gen extents;
    {
      typedef typename array::index index;
      array a(extents[4][5][6], so3);
      modifier.modify(a);
      const index idx0 = a.index_bases()[0];
      const index idx1 = a.index_bases()[1];
      const index idx2 = a.index_bases()[2];

      typename array::template array_view<3>::type b = a[
        indices[range(idx0 + 1, idx0 + 3)]
               [range(idx1 + 1, idx1 + 4)]
               [range(idx2 + 1, idx2 + 5)]
      ];
      assign(b);

      typename array::template const_array_view<3>::type c = b;
      access(c, const_array_tag());
    }
  }
  return 0;
}

template <typename array_modifier>
int do_test_storage(const array_modifier& modifier) {
  do_configuration(
    boost::c_storage_order(), boost::c_storage_order(), modifier);
  do_configuration(
    boost::fortran_storage_order(), boost::fortran_storage_order(), modifier);

  std::size_t ordering[] = {2, 0, 1, 3};
  bool ascending[] = {false, true, true, true};
  do_configuration(
    boost::general_storage_order<3>(ordering, ascending),
    boost::general_storage_order<4>(ordering, ascending), modifier);

  return 0;
}

struct null_modifier {
  template <typename array_t>
  void modify(array_t&) const {}
};

struct set_index_base_modifier {
  template <typename array_t>
  void modify(array_t a) const {
#ifdef BOOST_NO_SFINAE
  typedef boost::multi_array_type::index index;
  a.reindex(index(1));
#else
  a.reindex(1);
#endif
  }
};

struct reindex_modifier {
  template <typename array_t>
  void modify(array_t& a) const {
    boost::array<int, 4> bases = {{1, 2, 3, 4}};
    a.reindex(bases);
  }
};

struct reshape_modifier {
  template <typename array_t>
  void modify(array_t& a) const {
    typedef typename array_t::size_type size_type;
    std::vector<size_type> old_shape(a.num_dimensions());
    std::vector<size_type> new_shape(a.num_dimensions());

    std::copy(a.shape(), a.shape() + a.num_dimensions(), old_shape.begin());
    std::copy(old_shape.rbegin(), old_shape.rend(), new_shape.begin());

    a.reshape(new_shape);
    a.reshape(old_shape);
  }
};

int do_test_generative() {
  do_test_storage(null_modifier());
  do_test_storage(set_index_base_modifier());
  do_test_storage(reindex_modifier());
  do_test_storage(reshape_modifier());

  return 0;
}
