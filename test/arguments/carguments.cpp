#include <iostream>

auto main(int argc, char** argv) -> decltype(0)
{
  std::cout << argc << '\n';

  if (argc > 1)
    for (int i=0; i<argc; ++i)
      std::cout << **(argv+i) << '\n';

  return 0;
}
