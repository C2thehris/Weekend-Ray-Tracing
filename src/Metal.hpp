#pragma once
#include "rtweekend.hpp"
#include "Material.hpp"

class Metal : public Material<double>
{
public:
  Metal(const Color<double> &color, double fuzziness) : Material<double>(color), fuzziness_(std::min(fuzziness, 1.0)) {}

  bool scatter(const Ray<double> &ray, const Point3<double> &contact, const Vec3<double> &normal, Ray<double> &next, Color<double> &attenuation) const override
  {
    Vec3<double> reflected = reflect(ray.direction().unit(), normal);
    next = Ray<double>(contact, reflected /*+ fuzziness_ * random_unit_vector()*/);
    attenuation = this->albedo_;
    return next.direction() * normal > 0;
  }

private:
  double fuzziness_;
};