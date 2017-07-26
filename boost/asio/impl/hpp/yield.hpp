#include "coroutine.hpp"

#ifndef reenter
# define reenter(c) BOOST_ASIO_CORO_REENTER(c)
#endif

#ifndef yield
# define yield BOOST_ASIO_CORO_YIELD
#endif

#ifndef fork
# define fork BOOST_ASIO_CORO_FORK
#endif
