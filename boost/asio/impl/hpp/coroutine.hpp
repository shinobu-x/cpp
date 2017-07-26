class coroutine_ref {}

class coroutine {
public:
  coroutine() : value_(0) {}

  bool is_child() const { return value_ < 0; }

  bool is_parent() const { return !is_child(); }

  bool is_complete() const { return value_ == -1; }

private:
  friend class coroutine_ref;
  int value_;
};

#define BOOST_ASIO_CORO_REENTER(c) \
  switch (coroutine_ref _coro_value = c) \
    case -1: if (_coro_value) \
    { \
      goto terminate_coroutine; \
      terminate_coroutine: \
      _coro_value = 1; \
      goto bail_out_of_coroutine; \
      bail_out_of_cooutine: \
      break; \
    } \
    else case 0:
