#include <iostream>

template <typename T>
class basic_stopwatch : T {
public:
  explicit basic_stopwatch(bool start);
  explicit basic_stopwatch(char const* activity = "StopWatch",
    bool start = true);
  basic_stopwatch(std::ostream& log,
    char const* activity = "StopWatch",
    bool start = true);

  basic_stopwatch();
  unsigned lap_get() const;
  bool is_started() const;
  unsigned show(char const* event = "Show");
  unsigned start(char const* event = "Start");
  unsigned stop(char const* event = "Stop");

private:
  char const* activity_;
  unsigned lap_;
  std::ostream& log_;
};
