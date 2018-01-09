#add_subdirectory(mon)
#add_subdirectory(osd)
#add_subdirectory(os)
add_subdirectory(common)

# test
add_executable(
  sk_test
  test.cc
)
target_link_libraries(
  sk_test
  global
  ${BLKID_LIBRARIES}
)
