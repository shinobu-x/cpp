#include "../impl/hpp/coroutine.hpp"

#include "../impl/hpp/yield.hpp"

void yield_break_coro(coroutine& coro) {
  reenter (coro) {
//    yield return;
//    yield break;
  }
}

auto main() -> decltype(0) {
  return 0;
}
