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
