#pragma once
#include <optional>

#include "rtweekend.hpp"

struct Collision_t;

template <class T>
class Material
{
public:
  Material(const Color<T> &color) : albedo_(color) {}

  virtual bool scatter(const Ray<T> &ray, const Point3<T> &contact, const Vec3<T> &normal, Ray<T> &next, Color<T> &attenuation) const = 0;
  constexpr const Color<T> attenuation() const noexcept
  {
    return this->albedo_;
  }

  Vec3<double> reflect(const Vec3<double> &v, const Vec3<double> &n) const
  {
    return v - 2 * (v * n) * n;
  }

protected:
  Color<T> albedo_;
};