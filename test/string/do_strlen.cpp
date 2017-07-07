#include <cstdlib>  // size_t
#include <iostream>

void do_strlen_v1(char* s) {
  size_t l = 0;
  while (*(s++))
    l++;
  std::cout << l << '\n';
}
/*
void do_strlen_v2(const char** s) {
  size_t l = 0;
  while (**(s++))
    l++; 
  std::cout << **s << '\n';
  std::cout << l << '\n';
}
*/
void do_strlen_v3(const char* s) {
  size_t l = 0;
  while (*(s++))
    l++;
  std::cout << l << '\n';
}

void do_strlen_v4(char s[]) {
  size_t l = 0;
  while (*(s++))
    l++;
  std::cout << l << '\n';
}

auto main() -> decltype(0)
{
  char s1[] = "ABC";
  const char* s2[] = {"ABC"};
  const char s3[] = "ABC";
  do_strlen_v1(s1);
//  do_strlen_v2(s2);
  do_strlen_v3(s3);
  do_strlen_v4(s1);
  return 0;
}
