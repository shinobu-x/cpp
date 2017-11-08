# test
add_executable(
  sk_test_bluestore
  test.cc
)
# bluestore_types
add_executable(
  test_bluestore_types
  test_bluestore_types.cc
)
target_link_libraries(
  test_bluestore_types
  os
  global
)
