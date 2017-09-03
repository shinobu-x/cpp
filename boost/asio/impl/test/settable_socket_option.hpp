#include <cstddef>

template <typename pointer_type>
class settable_socket_option {
public:
  template <typename protocol>
  int level(const protocol&) const {
    return 0;
  }

  template <typename protocol>
  int name(const protocol&) const {
    return 0;
  }

  template <typename protocol>
  const pointer_type* data(const protocol&) const {
    return 0;
  }

  template <typename protocol>
  std::size_t size(const protocol&) const {
    return 0;
  }
};
