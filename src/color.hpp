#pragma once
#include <sstream>
#include <string>

#include "constants.hpp"
#include "vec3.hpp"

template <class T>
std::string write_color(const Color<T> &color) noexcept {
  std::stringstream out;
  out << static_cast<int>(color.x() * MAX_COLOR) << ' '
      << static_cast<int>(color.y() * MAX_COLOR) << ' '
      << static_cast<int>(color.z() * MAX_COLOR);
  return out.str();
}