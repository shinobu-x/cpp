# test
add_executable(
  sk_test_common
  test.cc
)
target_link_libraries(
  sk_test_common
  global
  ${BLKID_LIBRARIES}
)

# clock
add_executable(
  sk_test_clock
  sk_test_clock.cc
)
target_link_libraries(
  sk_test_clock
  global
  ${BLKID_LIBRARIES}
)

# ceph_clock
add_executable(
  sk_test_ceph_clock
  sk_test_ceph_clock.cc
)
target_link_libraries(
  sk_test_ceph_clock
  global
  ${BLKID_LIBRARIES}
)

# buffer
add_executable(
  sk_test_buffer
  sk_test_buffer.cc
)
target_link_libraries(
  sk_test_buffer
  global
  ${BLKID_LIBRARIES}
)
