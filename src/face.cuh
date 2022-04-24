#pragma once

#include "rtweekend_cuda.cuh"
#include "ray.cuh"

class Face
{
  color3 color_;
  MaterialProperty mat_;

  point3 center_;
  vec3 normal_;
  double d_;

public:
  Face(point3 center, MaterialProperty mat, color3 color, vec3 normal) noexcept : center_(center), color_(color), mat_(mat), normal_(normal)
  {
    d_ = normal * center;
  }
  __device__ MaterialProperty material() { return this->mat_; }

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
