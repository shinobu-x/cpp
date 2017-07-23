#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_STD_ADDRESSOF)
# include <memory>
#else
# include <boost/utility/addressof.hpp>
#endif

#if defined(BOOST_ASIO_HAS_STD_ADDRESSOF)
using std::addressof;
#else
using boost::addressof;
#endif
