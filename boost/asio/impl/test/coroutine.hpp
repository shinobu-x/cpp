#include <cassert>

#include "../hpp/coroutine.hpp"
#include "../hpp/yield.hpp"

void yield_break_coro(coroutine& coro) {
  reenter (coro) {
    yield return;
    yield break;
  }
}

void test_1() {
  coroutine coro;
  assert(!coro.is_complete());
  yield_break_coro(coro);
  assert(!coro.is_complete());
  yield_break_coro(coro);
  assert(coro.is_complete());
}

void return_coro(coroutine& coro) {
  reenter(coro) {
    return;
  }
}

void test_2() {
  coroutine coro;
  return_coro(coro);
  assert(coro.is_complete());
}

void exception_coro(coroutine& coro) {
  reenter(coro) {
    throw 1;
  }
}

void test_3() {
  coroutine coro;
  try { exception_coro(coro); } catch (int) {}
  assert(coro.is_complete());
}

void fall_off_end_coro(coroutine& coro) {
  reenter(coro) {}
}

void test_4() {
  coroutine coro;
  fall_off_end_coro(coro);
  assert(coro.is_complete());
}

auto main() -> decltype(0) {
  test_1(); test_2(); test_3(); test_4();
  return 0;
}  
