#pragma once

#include "rtweekend_cuda.cuh"

class Ray
{
public:
  __device__ Ray() noexcept {}
  __device__ Ray(const double3 &origin, const double3 &direction) noexcept
      : origin_(origin), direction_(direction) {}

  __device__ point3 origin() const noexcept { return this->origin_; }
  __device__ vec3 direction() const noexcept
  {
    return this->direction_;
  }

  __device__ point3 at(double time) const noexcept
  {
    return this->origin_ + this->direction_ * time;
  }

private:
  point3 origin_;
  vec3 direction_;
};