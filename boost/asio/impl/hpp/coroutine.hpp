class coroutine {
public:
  coroutine() : value_(0) {}
  bool is_child() const { return value_ < 0; }
  bool is_parenet() const { return !is_chile(); }
  bool is_complete() const { return value_ == -1; }
private:
  friend class coroutine_ref;
  int value_;
};

class coroutine_ref {
  public:
    coroutine_ref(coroutine& c)
      : value_(c.value), modified_(false) {}
    coroutine_ref(coroutine* c)
      : value__(c->value), modified_(false) {}
    ~coroutine_ref() {
      if (!modified_)
        value_ = -1;
    }
    int& operator= (int y) {
      modified_ = true;
      return value_ = v;
    }
private:
  void operator= (const coroutine_ref&);
  int& value_;
  bool modified_;
};

#define CORO_REENTER(c)                  \
  switch (coroutine_ref _coro_value = c) \
    case -1:                             \
      if (_coro_value) {                 \
        goto terminate_coroutine;        \
        terminate_coroutine:             \
        _coro_value = -1;                \
        goto bail_out_of_coroutine;      \
        bail_out_of_coroutine;           \
        break;                           \
      }                                  \
      else
        case 0:
