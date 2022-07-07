#pragma once

#include "rtweekend_cuda.cuh"
#include "sphere.cuh"
#include "face.cuh"
#include "material.cuh"

namespace RTW
{
  struct Collision_t
  {
    double t = INF;
    point3 contact;
    vec3 normal;
    Material *mat;
  };

  class ShapeContainer
  {
  private:
    Shape **_shapes;
    int _shapeCount;

  public:
    __device__ ShapeContainer(Shape **shapes, int shapeCount) : _shapes(shapes), _shapeCount(shapeCount) {}

    __device__ bool collision(const Ray &rin, Collision_t &closest, curandState_t *state) const
    {
      bool hit = false;
      for (int i = 0; i < _shapeCount; i += 1)
      {
        vec3 normal;
        double t = _shapes[i]->hit(rin, normal);
        if (t != -1 && t < closest.t)
        {
          hit = true;
          closest.t = t;
          closest.contact = rin.at(t);
          closest.normal = normal;
          closest.mat = _shapes[i]->material();
        }
      }
      return hit;
    }

    __device__ ~ShapeContainer()
    {
      delete *_shapes;
    }
  }; // class ShapeContainer

} // namespace RTW