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
