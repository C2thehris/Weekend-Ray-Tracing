#pragma once
#include "rtweekend.hpp"

class Camera
{
  Point3<double> origin;
  Point3<double> lower_left;
  Vec3<double> horizontal;
  Vec3<double> vertical;

public:
  constexpr Camera(int width, int height) noexcept
  {
    double aspect_ratio = static_cast<double>(width) / height;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * aspect_ratio;
    double focal_length = 1.0;

    origin = Point3<double>(0, 0, 0);
    horizontal = Point3<double>(viewport_width, 0, 0);
    vertical = Point3<double>(0, viewport_height, 0);
    lower_left = origin - horizontal / 2 - vertical / 2 -
                 Vec3<double>(0, 0, focal_length);
  }

  constexpr Ray<double> getRay(double u, double v) const noexcept
  {
    return Ray<double>(origin,
                       lower_left + u * horizontal + v * vertical - origin);
  }
};