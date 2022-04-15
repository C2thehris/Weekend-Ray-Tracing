#pragma once
#include "rtweekend.hpp"

class Camera
{
  Point3<double> origin;
  Point3<double> lower_left;
  Vec3<double> horizontal;
  Vec3<double> vertical;

public:
  constexpr Camera(const Point3<double> &lookFrom, const Point3<double> &lookTo, const Vec3<double> &vup, double vfov, double aspect_ratio) noexcept
  {
    double theta = degrees_to_radians(vfov);
    double h = std::tan(theta / 2);
    double viewport_height = 2.0 * h;
    double viewport_width = viewport_height * aspect_ratio;

    auto w = (lookFrom - lookTo).unit();
    auto u = crossProuduct(vup, w);
    auto v = crossProuduct(w, u);

    origin = lookFrom;
    horizontal = viewport_width * u;
    vertical = viewport_height * v;
    lower_left = origin - horizontal / 2 - vertical / 2 - w;
  }

  constexpr Ray<double> getRay(double s, double t) const noexcept
  {
    return Ray<double>(origin, lower_left + s * horizontal + t * vertical - origin);
  }
};