struct linux {};
struct windows {};
struct macosx {};  // Linux

#if __linux
typedef linux platform_type;
#elif _WIN32
typedef windows platform_type;
#elif __APPLE__
typedef macosx platform_type;
#endif

#define OUT(x) std::cout << "I am " #x << '\n';
