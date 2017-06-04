template <typename T>
class larger_than {
  T body_[2];
};

template <typename T>
T doit() {
  typedef char no_type;
  typedef larger_than<no_type> yes_type;
  yes_type a;
  no_type b;
}

auto main() -> int
{
  doit<char>();
  return 0;
}
