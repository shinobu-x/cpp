#include "../hpp/transporter.hpp"

#include <cstdlib>

template <typename T>
wrapper_base* wrapper<T>::clone() const {
  return new wrapper<T>(obj_);
}

template <typename T>
size_t wrapper<T>::size() const {
  return obj_.size();
}

transporter::~transporter() { delete ptr_; }

transporter::transporter(const transporter& that)
  : ptr_(that.ptr_ ? that.ptr_->clone() : 0) {}

transporter::transporter() : ptr_(0) {}

template <typename T>
transporter::transporter(const T& that)
  : ptr_(new wrapper<T>(that)) {}
