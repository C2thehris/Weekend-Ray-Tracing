#pragma once

#include "rtweekend_cuda.cuh"
#include "material.cuh"

namespace RTW
{
  class Shape
  {
  private:
    Material *_mat;

  public:
    __device__ Shape(Material *mat) : _mat(mat) {}

    __device__ Material *material() { return this->_mat; }

    __device__ virtual double hit(Ray in, vec3 &Nout) const = 0;

  }; // class Shape

} // namespace RTW