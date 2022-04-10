#pragma once
#include "vec3.hpp"

template <class T>
class Ray
{
public:
  constexpr Ray() noexcept {}
  constexpr Ray(const Point3<T> &origin, const Vec3<T> &direction) noexcept
      : origin_(origin), direction_(direction) {}

  constexpr const Point3<T> origin() const noexcept { return this->origin_; }
  constexpr const Vec3<T> direction() const noexcept
  {
    return this->direction_;
  }

  constexpr Point3<T> at(T time) const noexcept
  {
    return this->origin_ + direction_ * time;
  }

private:
  Point3<T> origin_;
  Vec3<T> direction_;
};