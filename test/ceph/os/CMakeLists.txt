add_subdirectory(bluestore)

# test
add_executable(
  sk_test_os
  test.cc
)
target_link_libraries(
  sk_test_os
  global
  ${BLKID_LIBRARIES}
)
