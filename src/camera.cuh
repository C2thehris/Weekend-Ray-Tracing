#pragma once
#include <cmath>

#include "rtweekend_cuda.cuh"
#include "ray.cuh"

class Camera
{
  point3 origin;
  point3 lower_left;
  vec3 horizontal;
  vec3 vertical;

public:
  __device__ Camera(point3 lookFrom, point3 lookTo, vec3 vup,
                    double vfov, double aspect_ratio) noexcept
  {
    double theta = degrees_to_radians(vfov);
    double h = std::tan(theta / 2);
    double viewport_height = 2.0 * h;
    double viewport_width = viewport_height * aspect_ratio;

    auto w = unit(lookFrom - lookTo);
    auto u = unit(crossProuduct(vup, w));
    auto v = crossProuduct(w, u);

    origin = lookFrom;
    horizontal = viewport_width * u;
    vertical = viewport_height * v;
    lower_left = origin - horizontal / 2 - vertical / 2 - w;
  }

  __device__ Ray getRay(double s, double t) const noexcept
  {
    return Ray(origin, lower_left + s * horizontal + t * vertical - origin);
  }
};