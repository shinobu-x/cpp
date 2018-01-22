#define BOOST_THREAD_FUTURE_BLOCKING                                           
#define BOOST_THREAD_FUTURE_USES_OPTIONAL
#define BOOST_THREAD_PROVIDES_EXECUTORS
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#define BOOST_THREAD_PROVIDES_FUTURE_CTOR_ALLOCATORS
#define BOOST_THREAD_PROVIDES_FUTURE_UNWRAP
#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#define BOOST_THREAD_USES_CHRONO
#define BOOST_THREAD_USES_MOVE
 
#include <boost/thread/detail/config.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/detail/move.hpp>
#include <boost/thread/detail/invoker.hpp>
#include <boost/thread/detail/invoke.hpp>
#include <boost/thread/detail/is_convertible.hpp>
#include <boost/thread/exceptional_ptr.hpp>
#include <boost/thread/futures/future_error.hpp>
#include <boost/thread/futures/future_error_code.hpp>
#include <boost/thread/futures/future_status.hpp>
#include <boost/thread/futures/is_future_type.hpp>
#include <boost/thread/futures/launch.hpp>
#include <boost/thread/futures/wait_for_all.hpp>
#include <boost/thread/futures/wait_for_any.hpp>
#include <boost/thread/lock_algorithms.hpp>
#include <boost/thread/lock_types.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread_only.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/thread/executor.hpp>
#include <boost/thread/executors/generic_executor_ref.hpp>
#include <boost/optional.hpp>
#include <boost/assert.hpp>
#include <boost/bind.hpp>
#include <boost/chrono/system_clocks.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/core/ref.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/function.hpp>
#include <boost/next_prior.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/decay.hpp>
#include <boost/type_traits/is_copy_constructible.hpp>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/thread/detail/memory.hpp>
#include <boost/container/scoped_allocator.hpp>
#include <boost/thread/csbl/tuple.hpp>
#include <boost/thread/csbl/vector.hpp>
#include <algorithm>
#include <list>
#include <vector>
#include <utility>
 
#if defined BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_FUTURE future
#else
#define BOOST_THREAD_FUTURE unique_future
#endif
