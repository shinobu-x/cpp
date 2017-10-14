#include <boost/thread/detail/platform.hpp>

#if defined(BOOST_THREAD_PLATFORM_WIN32)
#include <boost/thread/win32/thread_data.hpp>
#elif defined(BOOST_THREAD_PLATFORM_PTHREAD)
#include <boost/thread/pthread/thread_data.hpp>
#else
#error "Boost threads unavailable on this platform"
#endif

#include <boost/thread/detail/thread.hpp>
#if defined BOOST_THRED_PROVIDES_INTERRUPTIONS
#include <boost/thread/detail/thread_interruption.hpp>
#endif
#include <boost/thread/v2/thread.hpp>
