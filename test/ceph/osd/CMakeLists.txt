# test
add_executable(
  sk_test_osd
  test.cc
)
target_link_libraries(
  sk_test_osd
  global
  ${BLKID_LIBRARIES}
)

# osd_types
add_executable(
  test_osd_types
  test_osd_types.cc
)
add_ceph_unittest(
  test_osd_types
)
target_link_libraries(
  test_osd_types
  global
  ${BLKID_LIBRARIES}
)

# PGLog
add_executable(
  test_pglog
  test_pglog.cc
#  $<TARGET_OBJECTS:unit-main>
  $<TARGET_OBJECTS:store_test_fixture>
)
add_ceph_unittest(
  test_pglog
)
target_link_libraries(
  test_pglog
  osd
  os
  global
  ${CMAKE_DL_LIBS}
  ${BLKID_LIBRARIES}
)
