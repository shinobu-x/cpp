#include "include/memory.h"
#include <limits.h>
#include <errno.h>
#include <sys/uio.h>

#include "include/buffer.h"
#include "include/utime.h"
#include "include/coredumpctl.h"
#include "include/encoding.h"
#include "common/environment.h"
#include "common/Clock.h"
#include "common/safe_io.h"

#include "stdlib.h"
#include "fcntl.h"
#include "sys/stat.h"
#include "include/crc32c.h"
#include "common/sctp_crc32.h"

#define MAX_TEST 1000000
#define FILE_NAME "test_buffer"

static char cmd[128];

void test_1() {
  bool buffer_track = get_env_bool("CEPH_BUFFER_TRACK");
  unsigned len = 17;
  uint64_t history_alloc_bytes = 0;
  uint64_t history_alloc_num = 0;

  if (buffer_track)
    assert(buffer::get_total_alloc() == 0);

  { // create
    bufferptr bp(buffer::create(len));
    history_alloc_bytes += len;
    history_alloc_num++;
    assert(bp.length() == len);

    if (buffer_track) {
      assert(len == (unsigned)buffer::get_total_alloc());
      assert(history_alloc_bytes == buffer::get_history_alloc_bytes());
      assert(history_alloc_num == buffer::get_history_alloc_num());
    }
  }

  if (buffer_track)
    assert(buffer::get_total_alloc() == 0);

  { // claim_char
    char* s = new char[len];
    memset(s, 'X', len);
    bufferptr bp(buffer::claim_char(len, s));
    if (buffer_track) {
      assert(len == (unsigned)buffer::get_total_alloc());
      assert(history_alloc_bytes == buffer::get_history_alloc_bytes());
      assert(history_alloc_num == buffer::get_history_alloc_num());
    }

    assert(len == bp.length());
    assert(s = bp.c_str());
    bufferptr clone = bp.clone();
    history_alloc_bytes += len;
    history_alloc_num++;
    assert(0 == memcmp(clone.c_str(), bp.c_str(), len));
  }

  { // create_static
    char* s = new char[len];
    bufferptr bp(buffer::create_static(len, s));
    if (buffer_track) {
      assert(0 == buffer::get_total_alloc());
      assert(history_alloc_bytes == buffer::get_history_alloc_bytes());
      assert(history_alloc_num == buffer::get_history_alloc_num());
    }

    assert(len == bp.length());
    assert(s == bp.c_str());
  }

  { // create_malloc
    bufferptr bp(buffer::create_malloc(len));
    history_alloc_bytes += len;
    history_alloc_num++;
    if (buffer_track) {
      assert(len == (unsigned)buffer::get_total_alloc());
      assert(history_alloc_bytes == buffer::get_history_alloc_bytes());
      assert(history_alloc_num == buffer::get_history_alloc_num());
    }

    assert(len == bp.length());
  }
}

auto main() -> decltype(0) {
  test_1();
  return 0;
}
