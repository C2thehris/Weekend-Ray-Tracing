#pragma once

#include "rtweekend_cuda.cuh"
#include "shape.cuh"

namespace RTW
{
  class Face : public Shape
  {
    point3 center_;
    vec3 normal_;
    double d_;

  public:
    __device__ Face(point3 center, Material *mat, vec3 normal) noexcept : Shape(mat), center_(center), normal_(normal)
    {
      d_ = normal * center;
    }

    __device__ double hit(Ray in, vec3 &Nout) const
    {
      double t = (d_ - (normal_ * in.origin())) / (normal_ * in.direction());
      Nout = -normal_;
      return t > 0.001 && t <= 100 ? t : -1;
    }

    point3 getCenter() const { return center_; }
    vec3 getNormal() const { return normal_; }
  }; // class Face

} // namespace RTW
