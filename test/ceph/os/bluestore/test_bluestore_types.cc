#include "os/bluestore/bluestore_types.h"

template <typename T>
void test_instances(T& o) {
  std::list<T*> l;
  o.generate_test_instances(l);
}

void test_1() {
  { // bluestore_bdev_label_t
    bluestore_bdev_label_t o;
    test_instances(o);
  }
  { // bluestore_cnode_t
    bluestore_cnode_t o;
    test_instances(o);
  }
  { // AllocExtent
    AllocExtent o(0, 0);
  }
  { // ExtentList
    AllocExtentVector alloc;
    ExtentList o(&alloc, 0, 0);
  }
  { // bluestore_pextent_t
    bluestore_pextent_t o;
    test_instances(o);
  }
  { // denc_traits
    denc_traits<PExtentVector> o;
  }
  { // bluestore_extent_ref_map_t
    bluestore_extent_ref_map_t o;
    test_instances(o);
  }
  { // bluestore_blob_use_tracker_t
    bluestore_blob_use_tracker_t o;
    test_instances(o);
  }
  { // bluestore_blob_t
    bluestore_blob_t o;
    test_instances(o);
  }
  { // bluestore_shared_blob_t
    bluestore_shared_blob_t o(0);
    test_instances(o);
  }
  { // bluestore_onode_t
    bluestore_onode_t o;
    test_instances(o);
  }
  { // bluestore_deferred_op_t
    bluestore_deferred_op_t o;
    test_instances(o);
  }
  { // bluestore_deferred_transaction_t
    bluestore_deferred_transaction_t o;
    test_instances(o);
  }
}

auto main() -> decltype(0) {
  return 0;
}
