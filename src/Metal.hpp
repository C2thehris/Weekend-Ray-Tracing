#pragma once
#include "rtweekend.hpp"
#include "Material.hpp"

class Metal : public Material<double>
{
public:
  Metal(const Color<double> &color) : Material<double>(color) {}

  bool scatter(const Ray<double> &ray, const Point3<double> &contact, const Vec3<double> &normal, Ray<double> &next, Color<double> &attenuation) const override
  {
    Vec3<double> reflected = reflect(ray.direction().unit(), normal);
    next = Ray<double>(contact, reflected);
    attenuation = this->albedo_;
    return next.direction() * normal > 0;
  }

  Vec3<double> reflect(const Vec3<double> &v, const Vec3<double> &n) const
  {
    return v - 2 * (v * n) * n;
  }
};