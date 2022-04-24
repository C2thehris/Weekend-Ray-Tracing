#pragma once

#include "rtweekend_cuda.cuh"

class Sphere
{
private:
  double r_;
  color3 color_;
  point3 center_;

public:
  Sphere(point3 center, color3 color, double r) : center_(center), color_(color), r_(r) {}

  __device__ color3 color()
  {
    return this->color_;
  }

  __device__ double hit(Ray in, vec3 &Nout) const
  {
    auto RminusC = in.origin() - this->center_;
    double a = length_squared(in.direction());
    double halfb = in.direction() * RminusC;
    double c = length_squared(RminusC) - r_ * r_;

    auto discriminant = halfb * halfb - a * c;
    if (discriminant < 0)
      return -1;

    double sqrt_discriminant = sqrt(discriminant);
    double soonest = (-(halfb + sqrt_discriminant)) / a;
    if (soonest < 0.001)
      soonest = (-(halfb - sqrt_discriminant)) / a;
    if (soonest < 0.001)
      return -1;
    Nout = (in.at(soonest) - this->center_) / r_;
    return soonest;
  }
};