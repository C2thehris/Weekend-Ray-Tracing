#pragma once

#include "rtweekend_cuda.cuh"
#include "sphere.cuh"
#include "face.cuh"
#include "material.cuh"

struct Collision_t
{
  double t = INF;
  point3 contact;
  color3 color;
  vec3 normal;
  MaterialProperty mat;
  bool propogate;
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

      double t = spheres_[i].hit(rin, normal);
      if (t != -1 && t < closest.t)
      {
        hit = true;
        closest.t = t;
        closest.normal = normal;
        closest.contact = rin.at(t);
        closest.color = spheres_[i].color();
        closest.mat = spheres_[i].material();
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

      double t = faces_[i].hit(rin, normal);
      if (t != -1 && t < closest.t)
      {
        hit = true;
        closest.t = t;
        closest.contact = rin.at(t);
        closest.normal = normal;
        closest.color = faces_[i].color();
        closest.mat = faces_[i].material();
      }
    }
    return hit;
  }

public:
  ShapeContainer(Sphere *spheres, int n_spheres, Face *faces, int n_faces) : spheres_(spheres),
                                                                             sphere_count_(n_spheres), faces_(faces), face_count_(n_faces) {}

  __device__ bool collision(Ray &rin, Collision_t &closest, curandState_t *state) const
  {
    bool hit = false;
    bool ret = closestSphere(rin, closest);
    hit = ret || hit;
    ret = closestFace(rin, closest);
    hit = ret || hit;
    if (hit)
    {
      switch (closest.mat.type)
      {
      case LAMBERTIAN:
        closest.propogate = Lambertian::scatter(rin, closest.contact, closest.normal, state);
        break;
      case METAL:
        closest.propogate = Metal::scatter(rin, closest.contact, closest.normal, state, closest.mat.property);
        break;
      case DIELECTIC:
        closest.color = make_double3(1.0, 1.0, 1.0);
        closest.propogate = Dielectic::scatter(rin, closest.contact, closest.normal, state, closest.mat.property);
        break;
      default:
        printf("Unknown Material\n");
        closest.propogate = false;
        break;
      }
    }
    return hit;
  }

  ~ShapeContainer()
  {
    hostCheckError(cudaFree(spheres_));
    hostCheckError(cudaFree(faces_));
  }
};