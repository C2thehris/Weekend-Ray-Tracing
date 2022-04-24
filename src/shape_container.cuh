#pragma once

#include "rtweekend_cuda.cuh"
#include "sphere.cuh"
#include "face.cuh"

struct Collision_t
{
  double t = INF;
  color3 color;
  vec3 normal;
};

class ShapeContainer
{
  Sphere *spheres_;
  int sphere_count_;

  Face *faces_;
  int face_count_;

  __device__ bool closestSphere(Ray rin, Collision_t &closest) const
  {
    bool hit = false;
    for (int i = 0; i < sphere_count_; i += 1)
    {
      vec3 normal;

      int t = spheres_[i].hit(rin, normal);
      if (t != -1 && t < closest.t)
      {
        hit = true;
        closest.t = t;
        closest.normal = normal;
        closest.color = spheres_[i].color();
      }
    }
    return hit;
  }

  __device__ bool closestFace(Ray rin, Collision_t &closest) const
  {
    bool hit = false;
    for (int i = 0; i < face_count_; i += 1)
    {
      vec3 normal;

      int t = faces_[i].hit(rin, normal);
      if (t != -1 && t < closest.t)
      {
        hit = true;
        closest.t = t;
        closest.normal = normal;
        closest.color = faces_[i].color();
      }
    }
    return hit;
  }

public:
  ShapeContainer(Sphere *spheres, int n_spheres, Face *faces, int n_faces) : spheres_(spheres),
                                                                             sphere_count_(n_spheres), faces_(faces), face_count_(n_faces) {}

  __device__ bool collision(Ray rin, Collision_t &closest) const
  {
    bool hit = false;
    bool ret = closestSphere(rin, closest);
    hit = ret || hit;
    ret = closestFace(rin, closest);
    hit = ret || hit;
    return hit;
  }

  ~ShapeContainer()
  {
    hostCheckError(cudaFree(spheres_));
    hostCheckError(cudaFree(faces_));
  }
};