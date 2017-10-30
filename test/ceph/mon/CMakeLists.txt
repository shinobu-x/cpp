# test
add_executable(
  sk_test_mon
  test.cc
)
target_link_libraries(
  sk_test_mon
  global
  ${BLKID_LIBRARIES}
)
