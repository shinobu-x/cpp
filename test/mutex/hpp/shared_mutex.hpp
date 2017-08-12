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

      while (state.shared_count || state.exclusive) {
        state.exclusive_waiting_blocked = true;

        if (cv_status::timeout == exclusive_cond.wait_until(lk, abs_time)) {
          if (state.shared_count || state.exclusive) {
            state.exclusive_waiting_blocked = false;
            release_waiters();
            return false;
          }
          break;
        }
      }
      state.exclusive = true;

      return true;
    }
#endif

    bool try_lock() {
      boost::unique_lock<boost::mutex> lk(state_change);

      if (state.shared_count || state.exclusive)
        return false;
      else {
        state.exclusive = true;
        return true;
      }
    }

    void unlock() {
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_locked();
      state.exclusive = false;
      state.exclusive_waiting_blocked = false;
      state.assert_free();
      release_waiters();
    }

    void lock_upgrade() {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);

      while (state.exclusive || state.exclusive_waiting_blocked ||
        state.upgrade)
        shared_cond.wait(lk);

      state.lock_shared();
      state.upgrade = true;
    }

#if defined BOOST_THREAD_USES_DATETIME
    bool timed_lock_upgrade(boost::system_time const& timeout) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);
      while (state.exlusive || state.exclusive_waiting_blocked ||
        state.up)
        if (!shared_cond.timed_wait(lk, timeout)) {
          if (state.exclusive || state.exclusive_waiting_blocked ||
            state.upgrade)
              return false;
          break;
        }

      state.lock_shared();
      state.upgrade = true;

      return false;
    }

    template <typename TimeDuration>
    bool timed_lock_upgrade(TimeDuration const& relative_time) {
      return timed_lock_upgrade(boost::get_system_time() + relative_time);
    }
#endif
#ifdef BOOT_THREAD_USES_CHRONO
    template <class Rep, class Period>
    bool try_lock_upgrade_for(
      const boost::chrono::duration<Rep, Period>& rel_time) {
      return try_lock_upgrade_until(
        boost::chrono::steady_clock::now() + rel_time);
    }

    template <class Clock, class Duration>
    bool try_lock_upgrade_until(
      const boost::chrono::time_point<Clock, Duration>& abs_time) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
      boost::unique_lock<boost::mutex> lk(state_change);

      while (state.exclusive || state.exclusive_waiting_blocked ||
        state.upgrade)
        if (cv_status::timeout == shared_cond.wait_until(lk, abs_time)) {
          if (state.exclusive || state.exclusive_waiting_blocked ||
            state.upgrade)
            return false;
          break;
        }

      state.lock_shared();
      state.upgrade = true;

      return true;
    }
#endif

    bool try_lock_upgrade() {
      boost::unique_lock<boost::mutex> lk(state_change);

      if (state.exclusive || state.exclusive_waiting_blocked || state.upgrade)
        return false;
      else {
        state.lock_shared();
        state.upgrade = true;
        state.assert_lock_upgrade();
        return true;
      }
    }

    void unlock_upgrade() {
      boost::unique_lock<boost::mutex> lk(state_change);
      state.unlock_upgrade();

      if (!state.more_shared()) {
        state.exclusive_waiting_blocked = false;
        release_waiters();
      } else
        shared_cond.notify_all();
    }

    void unlock_upgrade_and_lock() {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_upgrade();
      state.unlock_shared();

      while (state.more_shared())
        upgrade_cond.wait(lk);

      state.upgrade = false;
      state.exclusive = true;
      state.assert_locked();
    }

    void unlock_and_lock_upgrade() {
      boost::unique_lock<boost::mutex> lk(state_chnange);
      state.assert_locked();
      state.exclusive = false;
      state.upgrade = true;
      state.lock_shared();
      state.exclusive_waiting_blocked = false;
      state.assert_lock_upgraded();
      release_waiters();
    }

    bool try_unlock_upgrade_and_lock() {
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_upgraded();

      if (!state.exclusive && !state.exclusive_waiting_blocked && 
        state.upgrade && state.shared == 1) {
        state.shared_count = 0;
        state.exclusive = true;
        state.upgrade = false;
        state.assert_locked();
        return true;
      }

      return false;
    }

#ifdef BOOST_THREAD_USES_CHRONO
    template <class Rep, class Period>
    bool try_unlock_upgrade_and_lock_for(
      bonst boost::chrono::duration<Rep, Period>& rel_time) {
      return try_unlock_upgrade_and_lock_until(
        boost::chrono::steady_clock::now + rel_time);
    }

    template <class Clock, class Duration>
    bool try_unlock_upgrade_and_lock_until(
      const boost::chrono::time_point<Clock, Duration>& abs_time) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);

      if (state.shared_count != 1)
        for (;;) {
          cv_status status = shared_cond.wait_until(lk, abs_time);

          if (state.shared_count == 1)
            break;
          if (status == cv_status::timeout)
            return false;
        }

      state.upgrade = false;
      state.exclusive = true;
      state.exclusive_waiting_blocked = false;
      state.shared_count = 0;

      return true;
    }
#endif

    void unlock_and_lock_shared() {
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_locked();
      state.exclusive = false;
      state.lock_shared();
      state.exclusive_waiting_blocked = false;
      release_waiters();
    }

#ifdef BOOST_THREAD_PROVIDES_SHARED_MUTEX_UPWARDS_CONVERSIONS
    bool try_unlock_shared_and_lock() {
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_shared();

      if (!state.exclusive && !state.exclusive_waiting_blocked &&
        !state.upgrade && state.shared_count == 1) {
        state.shared_count = 0;
        state.exclusive = true;
        return true;
      }

      return false;
    }

#ifdef BOOST_THREAD_USES_CHRONO
    template <class Rep, class Period>
    bool try_unlock_shared_and_lock_for(
      const boost::chrono::duration<Rep, Period>& rel_time) {
      return try_unlock_shared_and_lock_until(
        boost::chrono::steady_lock::now() + rel_time);
    }

    template <class Clock, class Duration>
    bool try_unlock_shared_and_lock_until(
      const boost::chrono::time_point<Clock, Duration>& abs_time) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_shared();

      if (state.shared_count != 1)
        for (;;) {
          cv_status status = shared_cond.wait_until(lk, abs_time);

          if (state.shared_count == 1)
            break;
          else
            return false;
        }

      state.upgrade = false;
      state.exclusive = true;
      state.exclusive_waiting_blocked = false;
      state.shared_count = 0;

      return true;
    }
#endif
#endif

    void unlock_upgrade_and_lock_shared() {
