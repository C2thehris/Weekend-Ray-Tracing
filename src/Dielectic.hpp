
#include "Material.hpp"

class Dielectic : public Material<double>
{
public:
  Dielectic(const double refraction) : Material<double>(Color<double>(1, 1, 1)), ir_(refraction) {}

  Vec3<double> refract(const Vec3<double> &uv, const Vec3<double> &n, double etai_over_etat) const
  {
    auto cos_theta = fmin(-uv * n, 1.0);
    Vec3<double> r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3<double> r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
  }

  bool scatter(const Ray<double> &ray, const Point3<double> &contact, const Vec3<double> &normal, Ray<double> &next, Color<double> &attenuation) const
  {
    attenuation = albedo_;
    double refraction_ratio;
    Vec3<double> correctedNormal = normal;
    if (normal * ray.direction() > 0)
    {
      correctedNormal = -normal;
      refraction_ratio = ir_;
    }
    else
    {
      refraction_ratio = (1.0 / ir_);
    }

    Vec3<double> unit_direction = ray.direction().unit();
    double cos_theta = fmin(-unit_direction * correctedNormal, 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    Vec3<double> direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
      direction = reflect(unit_direction, correctedNormal);
    else
      direction = refract(unit_direction, correctedNormal, refraction_ratio);

    next = Ray<double>(contact, direction);
    return true;
  }

private:
  double ir_;
  static double reflectance(double cosine, double ref_idx)
  {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
  }
};