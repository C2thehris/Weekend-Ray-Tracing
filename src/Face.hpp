#pragma once
#include "Shape.hpp"
#include "rtweekend.hpp"

template <class T>
class Face : public Shape<T>
{
public:
  Face(const Point3<T> &center, unique_ptr<Material<T>> &&material, const Vec3<T> &normal) : Shape<T>(center, move(material)), normal_(normal)
  {
    d_ = normal_ * center;
  }
  constexpr T collision(const Ray<T> &ray, Vec3<T> &N) const noexcept override
  {
    double t = (d_ - (normal_ * ray.origin())) / (normal_ * ray.direction());
    N = -normal_;
    return t > 0.001 && t <= 10 ? t : -1;
  }

private:
  Vec3<T> normal_;
  T d_;
};