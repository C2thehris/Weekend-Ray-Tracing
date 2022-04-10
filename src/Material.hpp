#pragma once
#include <optional>

#include "rtweekend.hpp"

struct Collision_t;

template <class T>
class Material
{
public:
  Material(const Color<T> &color) : albedo_(color) {}

  virtual Ray<T> scatter(const Ray<T> &ray, const Point3<double> &contact, const Vec3<T> &normal, Color<T> &attenuation) const = 0;
  constexpr const Color<double> attenuation() const noexcept
  {
    return this->albedo_;
  }

protected:
  Color<T> albedo_;
};