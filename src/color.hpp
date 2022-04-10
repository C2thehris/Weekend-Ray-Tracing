#pragma once
#include <sstream>
#include <string>

#include "rtweekend.hpp"
#include "vec3.hpp"

template <class T>
std::string write_color(const Color<T> &color) noexcept
{
  std::stringstream out;
  double r, g, b;
  r = std::sqrt(color.x() / SAMPLES);
  g = std::sqrt(color.y() / SAMPLES);
  b = std::sqrt(color.z() / SAMPLES);

  out << static_cast<int>(MAX_COLOR * clamp(r, 0, .9999)) << ' '
      << static_cast<int>(MAX_COLOR * clamp(g, 0, .9999)) << ' '
      << static_cast<int>(MAX_COLOR * clamp(b, 0, .9999));
  return out.str();
}