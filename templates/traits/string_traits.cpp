#include <cstdio>
#include <iostream>
#include <string>

template <typename string_t>
struct string_traits {
  typedef string_t const_iterator;
  typedef const string_t& argument_type;
 
  const_iterator begin(argument_type s);
  const_iterator end(argument_type s);

  static bool is_end_of_string(const_iterator i, argument_type s);
};

template <typename string_t>
void do_loop(const string_t& s) {
  typedef string_traits<string_t> traits_t;
  typename traits_t::const_iterator i = traits_t::begin(s);

  while (!traits_t::is_end_of_string(i, s))
    std::cout << *(i++);
}

template <typename char_t>
struct string_traits<std::basic_string<char_t> > {
  typedef char_t char_type;
  typedef typename std::basic_string<char_type>::const_iterator const_iterator;
  typedef const std::basic_string<char_type>& argument_type;

  static const_iterator begin(argument_type text) {
    return text.begin();
  }

  static const_iterator end(argument_type text) {
    return text.end();
  }

  static bool is_end_of_string(const_iterator i, argument_type s) {
    return i == s.end();
  }

};

template <>
struct string_traits<const char*> {
  typedef char char_type;
  typedef const char* const_iterator;
  typedef const char* argument_type;
  
  static const_iterator begin(argument_type text) {
    return text;
  }

  static const_iterator end(argument_type text) {
    return 0;
  }

  static bool is_end_of_string(const_iterator i, argument_type s) {
    return (i==0) || (*i==0);
  }
};

template <typename T>
T doit() {
  const char* c_string = "a b c d e f g";

//  for (char c = *c_string; c; c=*++c_string)

  string_traits<const char*> c_traits;
  std::cout << c_traits.begin(c_string) << '\n';
}

auto main() -> int
{
  doit<char>();
}
