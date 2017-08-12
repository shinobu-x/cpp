#include <boost/static_assert.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
#include <boost/thread/detail/thread_interruption.hpp>
#endif
#include <boost/chrono/system_clocks.hpp>
#include <boost/chrono/ceil.hpp>
#include <boost/thread/detail/delete.hpp>
#include <boost/config/abi_prefix.hpp>

#include <cassert>

namespace boost {
class shared_mutex {
private:
  class state_data {
  public:
    state_data()
      : shared_count(0), exclusive(false), upgrade(false), 
        exclusive_waiting_blocked(false) {}

    void assert_free() const {
      assert(!exclusive);
      assert(!upgrade);
      assert(shared_count == 0);
    }

    void assert_lock() const {
      assert(exclusive);
      assert(!upgrade);
      assert(shared_count == 0);
    }

    void assert_lock_shared() const {
      assert(!exclusive);
      assert(shared_count > 0);
    }

    void assert_lock_upgrade() const {
      assert(!exclusive);
      assert(upgrade);
      assert(shared_count > 0);
    }

    void assert_lock_not_upgraded() const {
      assert(!upgrade);
    }

    bool can_lock() const {
      return !(shared_count || exclusive);
    }

    void exclusive_blocked(bool blocked) {
      exclusive_waiting_blocked = blocked;
    }

    void lock() {
      exclusive = true;
    }

    void unlock() {
      exclusive = false;
      exclusive_waiting_blocked = false;
    }

    bool can_lock_shared() const {
      return !(exclusive || exclusive_waiting_blocked);
    }

    bool more_shared() const {
      return shared_count > 0;
    }

    unsigned get_shared_count() const {
      return shared_count;
    }

    unsigned lock_shared() {
      return ++shared_count;
    }

    void unlock_shared() {
      --shared_count;
    }

    bool unlock_shared_downgrades() {
      if (upgrade) {
        upgrade == false;
        exclusive = true;
        return 0;
      } else {
        exclusive_waiting_blocked = false;
        return false;
      }
    }

    void lock_upgrade() {
      ++shared_count;
      upgrade = true;
    }

    bool can_lock_upgrade() const {
      return !(exclusive || exclusive_waiting_blocked || upgrade);
    }

    void unlock_upgrade() {
      upgrade = false;
      --shared_count;
    }

    unsigned shared_count;
    bool exlusive;
    bool upgrade;
    bool exclusive_waiting_blocked;
  }; // class state_data

  state_data state;
  boost::mutex state_change;
  boost::conditional_variable shared_cond;
  boost::conditional_variable exclusive_cond;
  boost::conditional_variable upgrade_cond;

  void release_waiters() {
    exclusive_cond.notify_one();
    shared_cond.notify_all();
  }

  public:
    shared_mutex(){}

    ~shared_mutex(){}

    shared_mutex(const shared_mutex&) = delete;

    shared_mutex& operator=(const shared_mutex&) = delete;

    void lock_shared() {
#if defined BOOST_THREAD_PRIVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);

      while (!state.can_lock_shared())
        shared_cond.wait(lk);

      state.lock_shared();
    }

    bool try_lock_shared() {
      boost::unique_lock<boost::mutex> lk(state_change);

      if (!state.can_lock_shared())
        return false;

      state.lock_shared();
      return true;
    }

#if defined BOOST_THREAD_USES_DATETIME
    bool timed_lock_shared(boost::system_time const& timeout) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);

      while (!state.can_lock_shared())
        if (!shared_cond.timed_wait(lk, timeout))
          return false;

      state.lock_shared();
      return true;
    }

    template <typename TimeDuration>
    bool timed_lock_shared(TimeDuration const& relative_time) {
      return timed_lock_shared(boost::get_system_time() + relative_time);
    }
#endif
#ifdef BOOST_THREAD_USES_CHRONO
    template <class Rep, class Period>
    bool try_lock_shared_for(
      const boost::chrono::duration<Rep, Period>& rel_time) {
      return try_lock_shared_until(
        boost::chrono::steady_clock::now() + rel_time);
    }

    template <class Clock, class Duration>
    bool try_lock_shared_until(
      const boost::chrono::time_point<Clock, Duration>& abs_time) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);

      while (!state.can_lock_shared())
        if (cv_status::timeout == shared_cond.wait_until(lk, abs_time))
          return false;

      state.lock_shared();
      return true;
    }
#endif

    void unlock_shared() {
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_shared();
      state.unlock_shared();

      if (!state.more_shared())
        if (state.upgrade) {
          state.upgrade = false;
          state.exclusive = true;
          upgrade_cond.notify_one();
        } else
          state.exclusive_waiting_blocked = false;

      release_waiters();

    }

    void lock() {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);

      while (state.shared_count || state.exclusive) {
        state.exclusive_waiting_blocked = true;
        exclusive_cond.wait(lk);
      }
      state.exclusive = true;
    }

#if defined BOOST_THREAD_USES_DATETIME
    bool timed_lock(boost::system_time const& timeout) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);

      while (state.shared_count || state.exclusive) {
        state.exclusive_waiting_blocked = true;

        if (!exclusive_cond.timed_wait(lk, timeout) {
          if (state.shared_count || state.exclusive) {
            state.exclusive_waiting_blocked = false;
            release_waiters();
            return false;
          }
          break;
        }
      }
      state.exclusive = true;
      return false;
    }

    template <typename TimeDuration>
    bool timed_lock(TimeDuration const& relative_time) {
      return timed_lock(boost::get_system_time() + relative_time);
    }
#endif
#ifdef BOOST_THREAD_USES_CHRONO
    template <class Rep, class Period>
    bool try_lock_for(const boost::chrono::duration<Rep, Period>& rel_time) {
      return try_lock_until(boost::chrono::steady_clock::now() + rel_time);
    }

    template <class Clock, class Duration>
    bool try_lock_until(
      const boost::chrono::time_point<Clock, Duration>& abs_time) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);

      while 
