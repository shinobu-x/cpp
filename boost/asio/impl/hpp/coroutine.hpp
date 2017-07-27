class coroutine {
public:
  coroutine() : value_(0) {}
  bool is_child() const { return value_ < 0; }
  bool is_parenet() const { return !is_child(); }
  bool is_complete() const { return value_ == -1; }
private:
  friend class coroutine_ref;
  int value_;
};

class coroutine_ref {
  public:
    coroutine_ref(coroutine& c)
      : value_(c.value_), modified_(false) {}
    coroutine_ref(coroutine* c)
      : value_(c->value_), modified_(false) {}
    ~coroutine_ref() {
      if (!modified_)
        value_ = -1;
    }
    int& operator= (int v) {
      modified_ = true;
      return value_ = v;
    }
private:
  void operator= (const coroutine_ref&);
  int& value_;
  bool modified_;
};

#define CORO_REENTER(c)             \
  switch (coroutine_ref _coro_value = c)       \
    case -1:                                   \
      if (_coro_value) {                       \
        goto terminate_coroutine;              \
        terminate_coroutine:                   \
        _coro_value = -1;                      \
        goto bail_out_of_coroutine;            \
        bail_out_of_coroutine;                 \
        break;                                 \
      }                                        \
      else case 0:

#define CORO_YIELD_IMPL(n)                     \
  for (_coro_value = (n);;)                    \
    if (_coro_value == 0) {                    \
      case (n): ;                              \
      break;                                   \
    } else                                     \
      switch (_coro_value ? 0 : 1)             \
        for (;;)                               \
          case -1:                             \
            if (_coro_value)                   \
              goto terminate_coroutine;        \
            else                               \
              for (;;)                         \
          case 1:                              \
            if (_coro_value)                   \
              goto bail_out_of_coroutine;      \
            else case 0:

#define CORO_FORK_IMPL(n)                      \
  for (_coro_value = -(n);; _coro_value = (n)) \
    if (_coro_value == (n)) {                  \
      case -(n): ;                             \
      break;                                   \
    }                                          \
    else

#if defined(_MSC_VER)
# define CORO_YIELD CORO_YIELD_IMPL(__COUNTER__ + 1)
# define CORO_FORK CORO_FORK_IMPL(__COUNTER__ + 1)
#else
# define CORO_YIELD CORO_YIELD_IMPL(__LINE__)
# define CORO_FORK CORO_FORK_IMPL(__LINE__)
#endif

#pragma once
