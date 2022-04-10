
#include "vec3.hpp"
#include "Material.hpp"

class Lambertian : public Material<double>
{
public:
  Lambertian(const Color<double> &color) : Material<double>(color) {}

  Ray<double> scatter(const Ray<double> &ray, const Point3<double> &contact, const Vec3<double> &normal, Color<double> &attenuation) const override
  {
    Vec3<double> target = normal + random_unit_vector();
    attenuation = this->albedo_;
    return Ray<double>(contact, target - contact);
  }
};