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

# test_config
add_executable(
  test_config
  test_config.cc
)
target_link_libraries(
  test_config
#  ceph-common
  global
  ${BLKID_LIBRARIES}
#  ${EXTRALIBS}
)

# test_perf_counter
add_executable(
  test_perf_counters
  test_perf_counters.cc
)
target_link_libraries(
  test_perf_counters
  global
)
