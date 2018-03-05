namespace boost {
namespace future_state {

enum state {
  uninitialized,
  waiting,
  ready,
  moved
};

} // future_state

// boost
enum class future_errc {
  broken_promise,
  future_already_retrieved,
  promise_already_satisfied,
  no_state
};

enum class launch {
  none = unspecified,
  async = unspecified,
  deferred = unspecified,
  inherit = unspecified,
  any = async | deferred
};

enum class future_status {
  ready,
  timeout,
  deferred
};

namespace system {

template <>
struct is_error_code_enum<boost::future_errc> : public true_type {};

boost::system::error_code make_error_code(boost::future_errc);

boost::system::error_condition make_error_condition(boost::future_errc);

} // system

// boost
const system::error_category& future_category();

class future_error;

class exceptional_ptr;

template <typename R>
class promise;

template <typename R>
void swap(boost::promise<R>&, boost::promise<R>&) noexcept;

namespace container {

template <typename R, typename Allocator>
struct uses_allocator<boost::promise<R>, Allocator> : true_type {};

} // container

// boost
template <typename R>
class future;

template <typename R>
class shared_future;

template <typename R>
class packaged_task;

template <typename T>
struct is_future_type<boost::future<T> > : boost::true_type {};

template <typename T>
struct is_future_type<boost::shared_future<T> > :boost::true_type {}

template <typename R>
void swap(boost::packaged_task<R>&, boost::packaged_task<R>&) noexcept;

template <typename R, typename Allocator>
struct uses_allocator<boost::packaged_task<R>, Allocator> : true_type {};

template <typename F>
future<typename boost::result_of<
  typename boost::decay<F>::type()>::type> async(F);

template <typename R>
future<typename boost::result_of<
  typename boost::decay<F>::type()>::type> asycn(boost::launch, F);

template <typename F, typename... As>
future<typename boost::result_of<
  typename boost::decay<F>::type(
    typename boost::decay<As>::type...)>::type> async(F&&, As&&);

template <typename F, typename... As>
future<typename boost::result_of<
  typename boost::decay<F>::type(
    typename boost::decay<As>::type...)>::type> async(boost::luanch, F&&, As);

template <typename Ex, typename F, typename... As>
future<typename boost::result_of<
  typename boost::decay<F>::type(
    typename boost::decay<As>::type...)>::type> async(Ex&, F&&, As&&);

template <typename InputIter>
void wait_for_all(InputIter, InputIter);

template <typename F, typename... Fs>
void wait_for_all(F&, Fs&);

template <typename InputIter>
void wait_for_any(InputIter, InputIter);

template <typename F, typename... Fs>
void wait_for_any(F&, Fs&);

template <typename InputIter>
typename boost::disable_if<
  boost::is_future_type<InputIter>,
  boost::future<
    boost::csbl::vector<
      typename InputIter::value_type> > >::type when_all(InputIter, InputIter);

boost {
detail {

/* shared_state */
template <typename T>
struct shared_state :
  boost::detail::shared_state_base {}

template <typename T>
struct shared_state<T&> :
  boost::detail::shared_state_base {}

// storage_type
typedef boost::optional<T> storage_type;
typedef boost::csbl::unique_ptr<T> storage_type;
typedef T* storage_type;

// source_reference_type
typedef T const& source_reference_type;
typedef typename boost::conditional<
  boost::is_fundamental<T>::value,
  T,
  T const&>::type source_reference_type;
typedef T& source_reference_type;

// rvalue_source_type
typedef BOOST_THREAD_RV_REF(T) rvalue_source_type;
typedef typename boost::conditional<
  boost::thread_detail::is_convertible<
    T&,
    BOOST_THREAD_RV_REF(T),
    T const&>::type rvalue_source_type;

// move_dest_type
typedef T move_dest_type;
typedef typename boost::conditional<
  boost::thread_detail::is_convertible<
    T&,
    BOOST_THREAD_RV_REF(T)>::value,
  BOOST_THREAD_RV_REF(T),
  T>::type move_dest_type;
typedef T& move_dest_type;

// shared_future_get_result_type
typedef const T& shared_future_get_result_type;
typedef T& shared_future_get_result_type;

} // detail
} // boost

boost {
detail {

/* task_shared_state */
template <typename F, typename R, typename... As>
struct task_shared_state<F, R(As...)> :
  boost::detail::task_base_shared_state<R(As...)> {}

template <typename F, typename R>
struct task_shared_state<F, R()> :
  boost::detail::task_base_shared_state<R()> {}

template <typename F, typename R>
struct task_shared_state :
  boost::detail::task_base_shared_state<R> {}

// Call:
//   set_value_at_thread_exit
// or:
//   set_exception_at_thread_exit
void do_apply(BOOST_THREAD_RV_REF(As));
void do_apply();

// Call:
//   mark_finished_with_result
// or:
//   mark_exceptional_finish
void do_run(BOOST_THREAD_RV_REF(As));
void do_run()

/* task_base_shared_state */
template <typename R, typename... As>
struct task_base_shared_state<R(As...)> :
  boost::detail::shared_state<R> {}

template <typename R>
struct task_base_shared_state<R()> :
  boost::detail::shared_state<R> {}

template <typename R>
struct task_base_shared_state<R> :
  boost::detail::shared_state<R> {}

virtual void do_run(BOOST_THREAD_RV_REF(As)) = 0;
// Call
//   do_run(boost::move(as)...)
void run(BOOST_THREAD_RV_REF(As));

virtual void do_run() = 0;
// Call
//   do_run()
void run()

virtual void do_apply(BOOST_THREAD_RV_REF(As)) = 0;
// Call
//   do_apply(boost::move(as)...)
void apply(BOOST_THREAD_RV_REF(As));

virtual void do_apply() = 0;
// Call
//   do_apply()
void apply();

void owner_destroyed();

} // detail
} // boost
