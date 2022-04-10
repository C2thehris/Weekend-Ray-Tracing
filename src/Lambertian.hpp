
#include "vec3.hpp"
#include "Material.hpp"

class Lambertian : public Material<double>
{
public:
  Lambertian(const Color<double> &color) : Material<double>(color) {}

  bool scatter(const Ray<double> &ray, const Point3<double> &contact, const Vec3<double> &normal, Ray<double> &next, Color<double> &attenuation) const override
  {
    Vec3<double> target = normal + random_unit_vector();
    if (target.nearZero())
      target = normal;
    attenuation = this->albedo_;
    next = Ray<double>(contact, target);
    return true;
  }
};