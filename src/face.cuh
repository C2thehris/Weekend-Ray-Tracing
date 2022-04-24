#pragma once

#include "rtweekend_cuda.cuh"
#include "ray.cuh"

class Face
{
  point3 center_;
  color3 color_;
  vec3 normal_;
  double d_;

public:
  Face(point3 center, color3 color, vec3 normal) noexcept : center_(center), color_(color), normal_(normal)
  {
    d_ = normal * center;
  }

  __device__ color3 color() const noexcept
  {
    return this->color_;
  }

  __device__ double hit(Ray in, vec3 &Nout) const
  {
    double t = (d_ - (normal_ * in.origin())) / (normal_ * in.direction());
    Nout = -normal_;
    return t > 0.001 && t <= 100 ? t : -1;
  }
};
