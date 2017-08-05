#include <boost/thread/xtime.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>

#include "../macro/config.hpp"

inline boost::xtime delay(int, int, int); 
inline bool in_range(const boost::xtime&, int);

#pragma once
#include "utils.ipp"
