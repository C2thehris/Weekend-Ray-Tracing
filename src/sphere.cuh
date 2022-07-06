#pragma once

#include "rtweekend_cuda.cuh"
#include "shape.cuh"

namespace RTW
{

  class Sphere : public Shape
  {
  private:
    double r_;
    point3 center_;

  public:
    __device__ Sphere(point3 center, Material *mat, double r) : Shape(mat), center_(center), r_(r) {}

    __device__ double hit(Ray in, vec3 &Nout) const override
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

    double getR() const { return r_; }
    point3 getCenter() const { return center_; }

  }; // class Sphere

}