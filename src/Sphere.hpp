#pragma once
#include "Shape.hpp"

template <class T>
class Sphere : public Shape<T>
{
public:
  constexpr Sphere(const Vec3<T> &center, unique_ptr<Material<T>> &&material,
                   const T radius) noexcept
      : Shape<T>(center, move(material)), radius_(radius) {}

  constexpr T collision(const Ray<T> &r, Vec3<T> &N) const noexcept override
  {
    auto RminusC = r.origin() - this->center_;
    double a = r.direction().length_squared();
    double halfb = r.direction() * RminusC;
    double c = (RminusC.length_squared()) - (radius_ * radius_);

    double discriminant = halfb * halfb - a * c;
    if (discriminant < 0)
    {
      return -1;
    }
    double sqrtd = sqrt(discriminant);
    double soonest = (-halfb - sqrtd) / (a);
    if (soonest < 0)
      soonest = (-halfb + sqrtd) / a;
    if (soonest < 0)
      return -1;
    N = (r.at(soonest) - this->center_) / radius_;
    if (r.direction() * N > 0)
      N = -N;

    return soonest; // Just forward for now
  }

private:
  T radius_;
};