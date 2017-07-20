#include <cstdlib>

struct wrapper_base {
  virtual ~wrapper_base() {}
  virtual wrapper_base* clone() = 0;
  virtual size_t size() const = 0;
};

template <typename T>
struct wrapper : wrapper_base {
  T obj_;

  virtual wrapper_base* clone() const;
  virtual size_t size() const;
};

class transporter {
public:
  wrapper_base* ptr_;

public:
  ~transporter();
  transporter(const transporter&);
  transporter();

  template <typename T>
  transporter(const T&);
};
  
#pragma once
#include "../ipp/transporter.ipp"
