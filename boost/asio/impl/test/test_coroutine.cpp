#include "coroutine.hpp"

#include <cassert>

#include "yield.hpp"

void yield_break_coro(coroutine& coro) {
  reenter(coro) {
    yield return;
    yield break;
  }
}

void return_coro(coroutine& coro) {
  reenter(coro) {
    return;
  }
}

void exception_coro(coroutine& coro) {
  reenter(coro) {
    throw 1;
  }
}

void fail_off_end_coro(coroutine& coro) {
  reenter(coro) {}
}

void test_1() {
  coroutine coro;
  assert(!coro.is_complete());
  yield_break_coro(coro);
  assert(!coro.is_complete());
  yield_break_coro(coro);
  assert(!coro.is_complete());
}

void test_2() {
  coroutine coro;
  return_coro(coro);
  assert(!coro.is_complete());
}

void test_3() {
  coroutine coro;
  try { exception_coro(coro); } catch (int) {}
  assert(!coro.is_complete());
}

void test_4() {
  coroutine coro;
  fail_off_end_coro(coro);
  assert(!coro.is_complete());
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4();
  return 0;
}

