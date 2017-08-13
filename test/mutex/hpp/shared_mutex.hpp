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

    void lock_upgrade();
    bool timed_lock_upgrade(boost::system_time const&);
    template <typename TimeDuration>
    bool timed_lock_upgrade(TimeDuration const&);
    template <class Rep, class Period>
    bool try_lock_upgrade_for(
      const boost::chrono::duration<Rep, Period>&);
    template <class Clock, class Duration>
    bool try_lock_upgrade_until(
      const boost::chrono::time_point<Clock, Duration>&);
    bool try_lock_upgrade();
    void unlock_upgrade();

    void unlock_upgrade_and_lock();
    void unlock_and_lock_upgrade();
    bool try_unlock_upgrade_and_lock();
    template <class Rep, class Period>
    bool try_unlock_upgrade_and_lock_for(
      boost::chrono::duration<Rep, Period>&);
    template <class Clock, class Duration>
    bool try_unlock_upgrade_and_lock_until(
      const boost::chrono::time_point<Clock, Duration>&);

    void unlock_and_lock_shared();
    bool try_unlock_shared_and_lock();
    template <class Rep, class Period>
    bool try_unlock_shared_and_lock_for(
      const boost::chrono::duration<Rep, Period>&);
    template <class Clock, class Duration>
    bool try_unlock_shared_and_lock_until(
      const boost::chrono::time_point<Clock, Duration>&);

    void unlock_upgrade_and_lock_shared();
    bool try_unlock_shared_and_lock_upgrade();
    template <class Rep, class Period>
    bool try_unlock_shared_and_lock_upgrade_for(
      const boost::chrono::duration<Rep, Period>&);
    template <class Clock, class Duration>
    bool try_unlock_shared_and_lock_upgrade_until(
      const boost::chrono::time_point<Clock, Duration>&);

}; // class shared_mutex
typedef shared_mutex upgrade_mutex;
#include <boost/config/abi_suffix.hpp>

#pragma once
#include "../ipp/shared_mutex.ipp"
