#pragma once
#include <memory>

#include "ray.hpp"
#include "vec3.hpp"
#include "rtweekend.hpp"
#include "Material.hpp"

using std::unique_ptr;

template <class T>
class Shape
{
public:
  Shape(const Point3<T> &center, unique_ptr<Material<T>> &&material) : center_(center)
  {
    this->material_ = move(material);
  }

  constexpr virtual T collision(const Ray<T> &r, Vec3<T> &N) const noexcept = 0;
  constexpr const Point3<T> center() const noexcept
  {
    return this->center_;
  }
  constexpr const Material<T> *material() const noexcept
  {
    return this->material_.get();
  }

protected:
  Point3<T> center_;
  unique_ptr<Material<T>> material_;
};