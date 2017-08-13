#include "../hpp/shared_mutex.hpp"

shared_mutex::state_data::state_data()
  : shared_count(0), exclusive(false), upgrade(false), 
  exclusive_waiting_blocked(false) {}

void shared_mutex::state_data::assert_free() const {
  assert(!exclusive);
  assert(!upgrade);
  assert(shared_count == 0);
}

void shared_mutex::state_data::assert_locked() const {
  assert(exclusive);
  assert(!upgrade);
  assert(shared_count == 0);
}

void shared_mutex::state_data::assert_lock_shared() const {
  assert(!exclusive);
  assert(shared_count > 0);
}

void shared_mutex::state_data::assert_lock_upgrade() const {
  assert(!exclusive);
  assert(upgrade);
  assert(shared_count > 0);
}

void shared_mutex::state_data::assert_lock_not_upgraded() const {
  assert(!upgrade);
}

bool shared_mutex::state_data::can_lock() const {
  return !(shared_count || exclusive);
}

void shared_mutex::state_data::exclusive_blocked(bool blocked) {
  exclusive_waiting_blocked = blocked;
}

void shared_mutex::state_data::lock() {
  exclusive = true;
}

void shared_mutex::state_data::unlock() {
  exclusive = false;
  exclusive_waiting_blocked = false;
}

bool shared_mutex::state_data::can_lock_shared() const {
  return !(exclusive || exclusive_waiting_blocked);
}

bool shared_mutex::state_data::more_shared() const {
  return shared_count > 0;
}

unsigned shared_mutex::state_data::get_shared_count() const {
  return shared_count;
}

unsigned shared_mutex::state_data::lock_shared() {
  return ++shared_count;
}

unsigned shared_mutex::state_data::unlock_shared() {
  return --shared_count;
}

bool shared_mutex::state_data::unlock_shared_downgrades() {
  if (upgrade) {
    upgrade = false;
    exclusive = true;
    return 0;
  } else {
    exclusive_waiting_blocked = false;
    return false;
  }
}

void shared_mutex::state_data::lock_upgrade() {
  ++shared_count;
  upgrade = true;
}

bool shared_mutex::state_data::can_lock_upgrade() const {
  return !(exclusive || exclusive_waiting_blocked || upgrade);
}

void shared_mutex::state_data::unlock_upgrade() {
  upgrade = false;
  --shared_count;
}

shared_mutex::shared_mutex(){}

shared_mutex::~shared_mutex(){}

void shared_mutex::lock_shared() {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
  boost::this_thread::disable_interruption do_not_disturb;
#endif
  boost::unique_lock<boost::mutex> lk(state_change);

  while (!state.can_lock_shared())
    shared_cond.wait(lk);

  state.lock_shared();
}

bool shared_mutex::try_lock_shared() {
  boost::unique_lock<boost::mutex> lk(state_change);

  if (!state.can_lock_shared())
    return false;

  state.lock_shared();

  return true;
}

void shared_mutex::release_waiters() {
  exclusive_cond.notify_one();
  shared_cond.notify_all();
}

#if defined BOOST_THREAD_USES_DATETIME
bool shared_mutex::timed_lock_shared(boost::system_time const& timeout) {
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
bool shared_mutex::timed_lock_shared(TimeDuration const& relative_time) {
  return timed_lock_shared(boost::get_system_time() + relative_time);
}
#endif

#ifdef BOOST_THREAD_USES_CHRONO
template <class Rep, class Period>
bool shared_mutex::try_lock_shared_for(
  const boost::chrono::duration<Rep, Period>& rel_time) {
  return try_lock_shared_until(boost::chrono::steady_clock::now() + rel_time);
}

template <class Clock, class Duration>
bool shared_mutex::try_lock_shared_until(
  const boost::chrono::time_point<Clock, Duration>& abs_time) {
#if defined BOOST_THREAD_PROVIDES_INTERRUPTIONS
  boost::this_thread::disable_interruption do_not_disturb;
#endif
  boost::unique_lock<boost::mutex> lk(state_change);

  while (!state.can_lock_shared())
    if (boost::cv_status::timeout == shared_cond.wait_until(lk, abs_time))
      return false;

  state.lock_shared();

  return true;
}
#endif

void shared_mutex::unlock_shared() {
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

void shared_mutex::lock() {
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
