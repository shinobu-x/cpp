#include "coroutine.hpp"
#ifndef reenter
#define reenter(c) CORO_REENTER(c)
#endif

#ifndef yield
#define yield CORO_YIELD
#endif

#ifndef fork
#define fork CORO_FORK
#endif
