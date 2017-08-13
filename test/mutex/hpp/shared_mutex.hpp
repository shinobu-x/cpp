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

class shared_mutex {
private:
  class state_data {

  public:
    state_data();
    void assert_free() const;
    void assert_locked() const;
    void assert_lock_shared() const;
    void assert_lock_upgrade() const;
    void assert_lock_not_upgraded() const;
    bool can_lock() const;
    void exclusive_blocked(bool);
    void lock(); 
    void unlock();
    bool can_lock_shared() const;
    bool more_shared() const;
    unsigned get_shared_count() const;
    unsigned lock_shared();
    unsigned unlock_shared();
    bool unlock_shared_downgrades();
    void lock_upgrade();
    bool can_lock_upgrade() const;
    void unlock_upgrade();
    unsigned shared_count;
    bool exclusive;
    bool upgrade;
    bool exclusive_waiting_blocked;
  }; // class state_data

  state_data state;
  boost::mutex state_change;
  boost::condition_variable shared_cond;
  boost::condition_variable exclusive_cond;
  boost::condition_variable upgrade_cond;

  void release_waiters();

  public:
    shared_mutex();
    ~shared_mutex();
    shared_mutex(const shared_mutex&) = delete;
    shared_mutex& operator=(const shared_mutex&) = delete;
    shared_mutex(shared_mutex&&) = delete;
    shared_mutex& operator=(shared_mutex&&) = delete;

    void lock_shared();
    bool try_lock_shared();
    bool timed_lock_shared(boost::system_time const&);
    template <typename TimeDuration>
    bool timed_lock_shared(TimeDuration const&);
    template <class Rep, class Period>
    bool try_lock_shared_for(const boost::chrono::duration<Rep, Period>&);
    template <class Clock, class Duration>
    bool try_lock_shared_until(
      const boost::chrono::time_point<Clock, Duration>&);
    void unlock_shared();
    void lock();
    bool timed_lock(boost::system_time const&);
    template <typename TimeDuration>
    bool timed_lock(TimeDuration const&);
    template <class Rep, class Period>
    bool try_lock_for(const boost::chrono::duration<Rep, Period>&);
    template <class Clock, class Duration>
    bool try_lock_until(const boost::chrono::time_point<Clock, Duration>&);
    bool try_lock();
    void unlock();

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
      while (state.exclusive || state.exclusive_waiting_blocked ||
        state.upgrade)
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
#endif
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
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_locked();
      state.exclusive = false;
      state.upgrade = true;
      state.lock_shared();
      state.exclusive_waiting_blocked = false;
      state.assert_lock_upgrade();
      release_waiters();
    }

    bool try_unlock_upgrade_and_lock() {
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_upgrade();

      if (!state.exclusive && !state.exclusive_waiting_blocked && 
        state.upgrade && state.shared_count == 1) {
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
      boost::chrono::duration<Rep, Period>& rel_time) {
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
          boost::cv_status status = shared_cond.wait_until(lk, abs_time);

          if (state.shared_count == 1)
            break;
          if (status == boost::cv_status::timeout)
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
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_upgrade();
      state.upgrade = false;
      state.exclusive_waiting_blocked = false;
      release_waiters();
    }

#ifdef BOOST_THREAD_PROVIDES_SHARED_MUTEX_UPWARDS_CONVERSIONS
    bool try_unlock_shared_and_lock_upgrade() {
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_shared();

      if (!state.exclusive && !state.exclusive_waiting_blocked &&
        !state.exclusive_waiting_blocked && !state.upgrade) {
        state.upgrade = true;
        return true;
      }

      return false;
    }

#ifdef BOOST_THREAD_USES_CHRONO
    template <class Rep, class Period>
    bool try_unlock_shared_and_lock_upgrade_for(
      const boost::chrono::duration<Rep, Period>& rel_time) {
      return try_unlock_shared_and_lock_upgrade_until(
        boost::chrono::time_point<Clock, Duration>& abs_time)
    }

    template <class Clock, class Duration>
    bool try_unlock_shared_and_lock_upgrade_until(
      const boost::chrono::time_point<Clock, Duration>& abs_time) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
      boost::this_thread::disable_interruption do_not_disturb;
#endif
      boost::unique_lock<boost::mutex> lk(state_change);
      state.assert_lock_shared();

      if (state.exclusive || state.exclusive_waiting_blocked ||
        state.upgrade)
        for (;;) {
          cv_status status = exlusive_cond.wait_until(lk, abs_time);

          if (!state.exlusive && !state.exclusive_waiting_blocked &&
            !state.upgrade)
            break;
          if (status == cv_status::timeout)
           return false;
        }

      state.upgrade = true;

      return true;
    }
#endif
#endif
}; // class shared_mutex
typedef shared_mutex upgrade_mutex;
#include <boost/config/abi_suffix.hpp>

#pragma once
#include "../ipp/shared_mutex.ipp"
