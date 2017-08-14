#include <thread>
#include <utility>
#include <cassert>

void get_thread_id(std::thread::id* that_id) {
  *that_id = std::this_thread::get_id();
}

std::thread make_thread(std::thread::id* that_id) {
  std::thread t(get_thread_id, that_id);
  return std::move(t);
}

auto main() -> decltype(0) {
  std::thread::id this_id;
  std::thread that_thread = make_thread(&this_id);
  std::thread::id that_id = that_thread.get_id();
  that_thread.join();
  assert(this_id == that_id);
}
