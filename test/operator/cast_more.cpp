struct A {
  operator int () { return i; }
  operator char* () { return cp; }
private:
  int i;
  char* cp;
};

struct B : public A {
  operator float () { return f; }
  operator char* () { return cp; }
private:
  float f;
  char* cp;
};

auto main() -> decltype(0) {
  B b;
  char* cp = b;
}
