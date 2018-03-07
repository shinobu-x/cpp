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
//////////////////////////////////////
// shared_state                     //
//////////////////////////////////////
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

// Member variables
storage_type result_;

// Member methods
// Call:
//   mark_finished_internal;
void mark_finished_with_result_internal(
  source_reference_type,
  boost::unique_lock<boost::mutex>&);

void mark_finished_with_result_internal(
  rvalue_source_type,
  boost::unique_lock<boost::mutex>&);

template <typename... As>
void mark_finished_with_result_internal(
  boost::unique_lock<boost::mutex>&,
  BOOST_THREAD_FWD_REF(As));

// Call:
//   mark_finished_with_result_internal
void mark_finished_with_result(
  source_reference_type);

void mark_finished_with_result(
  rvalue_reference_type);

storage_type& get_storage(
  boost::unique_lock<boost::mutex>&);

virtual move_dest_type get(
  boost::unique_lock<boost::mutex>&);

move_dest_type get();

virtual shared_future_get_result_type get_result_type(
  boost::unique_lock<boost::mutex>&);

shared_future_get_result_tpe get_result_type();

void set_value_at_thread_exit(
  source_reference_type);

void set_value_at_thread_exit(
  rvalue_source_type);

// Private
shared_state(shared_state const&);
shared_state& operator=(shared_state const&);

} // detail
} // boost

boost {
detail {

//////////////////////////////////////
// task_shared_state                //
//////////////////////////////////////
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

//////////////////////////////////////
// task_base_shared_state           //
//////////////////////////////////////
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
// Call:
//   do_run(boost::move(as)...)
void run(BOOST_THREAD_RV_REF(As));

virtual void do_run() = 0;
// Call:
//   do_run()
void run()

virtual void do_apply(BOOST_THREAD_RV_REF(As)) = 0;
// Call:
//   do_apply(boost::move(as)...)
void apply(BOOST_THREAD_RV_REF(As));

virtual void do_apply() = 0;
// Call:
//   do_apply()
void apply();

void owner_destroyed();

} // detail
} // boost

boost {

//////////////////////////////////////
// packaged_task                    //
//////////////////////////////////////
template <typename R, typename... As>
class packaged_task<R(As...)> {}

template <typename R>
class packaged_task<R()> {}

template <typename R>
class packaged_task {}

// Member variables
typedef boost::shared_ptr<
  boost::detail::task_base_shared_state<R(As...)> > task_ptr;
typedef boost::shared_ptr<
  boost::detail::task_base_shared_state<R()> > task_ptr;
typedef boost::shared_ptr<
  boost::detail::task_base_shared_state<R> > task_ptr;

boost::shared_ptr<boost::detail::task_base_shared_state<R(As...)> > task_
boost::shared_ptr<boost::detail::task_base_shared_state<R()> > task_;
boost::shared_ptr<boost::detail::task_base_shared_state<R> > task_;

typedef R result_type;

// Member methods
void set_executor(executor_ptr_type);
void reset();
void swap(packaged_task& that) BOOST_NOEXCEPT;
bool valid() const BOOST_NOEXCEPT;
BOOST_THREAD_FUTURE<R> get_future();

// Ctor
explicit packaged_task(R(*f)(),BOOST_THREAD_FWD_REF(As)...);
explicit packaged_task(R(*f)());
explicit packaged_task(R(*f)());

template <typename F>
explicit packaged_task(
  BOOST_THREAD_FWD_REF(F) f,
  typename boost::disable_if<
    boost::is_same<
      typename boost::decay<F>::type,
      packaged_task>,
    dummy* >::type = 0);

template <typename F>
explicit packaged_task(
  F const& f,
  typename boost::disable_if<
    boost::is_same<
      typename boost::decay<F>::type,
      packaged_task>,
    dummy* >::type = 0);

template <typename F>
explicit packaged_task(
  BOOST_THREAD_RV_REF(F) f);

// Execution
// Call:
//   task_->run
void operator()(As...);
void operator()();

// Call:
//  task_->apply
void make_ready_at_thread_exit(As...);
void make_ready_at_thread_exit();

// Call:
//  task_->set_wait_callback
template <typename F>
void set_wait_callback(F);

} // boost
