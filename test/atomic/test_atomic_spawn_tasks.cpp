#include <boost/atomic.hpp>
#include <boost/chrono.hpp>
#include <boost/chrono/chrono_io.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

inline boost::chrono::high_resolution_clock::time_point real_time_now() {
  return boost::chrono::high_resolution_clock::now();
}

class task {
  boost::atomic<bool> _is_exiting;
  boost::chrono::high_resolution_clock::time_point _next_tick_time;
public:
  bool is_exiting() const {
    return _is_exiting;
  }

  boost::chrono::high_resolution_clock::time_point spawn_tasks() {

    const boost::chrono::high_resolution_clock::time_point now =
      real_time_now();

    if (_next_tick_time < now)
      _next_tick_time = now + boost::chrono::seconds(1);

    return _next_tick_time;
  }
};

auto main() -> decltype(0) {
  static const boost::chrono::milliseconds min_time_tasks_spawn_frequency =
    boost::chrono::milliseconds(1);

  boost::condition_variable task_spawn_condition;

  task t;

  boost::mutex main_thread_mutex;
  boost::unique_lock<boost::mutex> main_thread_lock(main_thread_mutex);

  int i = 11;

  while (--i) {
    const boost::chrono::high_resolution_clock::time_point next_task_spawn =
      t.spawn_tasks();

    const boost::chrono::high_resolution_clock::time_point now =
      real_time_now();

    const boost::chrono::high_resolution_clock::time_point next_min_spawn_time =
      now + min_time_tasks_spawn_frequency;

    const boost::chrono::high_resolution_clock::time_point next_spawn_time =
     ((next_task_spawn > boost::chrono::high_resolution_clock::time_point()) &&
       (next_task_spawn < next_min_spawn_time))
       ? next_task_spawn : next_min_spawn_time;

    const boost::chrono::high_resolution_clock::time_point::duration wait_time =
      next_spawn_time - now;

    if (wait_time > wait_time.zero()) {
      boost::this_thread::sleep_for(boost::chrono::seconds(1));
      std::cout << next_spawn_time << '\n';
      task_spawn_condition.wait_until(main_thread_lock, next_spawn_time);
    }
  }
}
