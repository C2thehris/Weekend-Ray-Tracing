#pragma once

#include "rtweekend_cuda.cuh"
#include "ray.cuh"

namespace RTW
{

  class Material
  {
  public:
    __device__ Material(color3 color) : color_(color) {}
    __device__ virtual bool scatter(Ray &ray, point3 contact, vec3 normal, curandState *state) const = 0;

    __device__ color3 color() const
    {
      return this->color_;
    }

  private:
    color3 color_;
  };

  class Lambertian : public Material
  {
  public:
    __device__ Lambertian(color3 color) : Material(color) {}
    __device__ bool scatter(Ray &ray, point3 contact, vec3 normal, curandState_t *state) const override
    {
      vec3 target = normal + random_unit_vector(state);
      if (nearZero(target))
        target = normal;
      ray = Ray(contact, target);
      return true;
    }
  };

  class Metal : public Material
  {
  public:
    __device__ Metal(color3 color, double fuzziness) : Material(color), fuzziness_(fuzziness) {}

    __device__ bool scatter(Ray &ray, point3 contact, vec3 normal, curandState_t *state) const override
    {
      vec3 reflected = reflect(unit(ray.direction()), normal);
      ray = Ray(contact, reflected /*+ this->fuzziness_ * random_unit_vector(state)*/);
      return ray.direction() * normal > 0;
    }

  private:
    double fuzziness_;
  };

  class Dielectic : public Material
  {
  public:
    __device__ Dielectic(color3 color, float ir) : Material(color), ir_(ir) {}

    __device__ bool scatter(Ray &ray, point3 contact, vec3 normal, curandState_t *state) const override
    {
      double refraction_ratio;
      vec3 correctedNormal = normal;
      if (normal * ray.direction() > 0)
      {
        correctedNormal = -normal;
        refraction_ratio = this->ir_;
      }
      else
      {
        refraction_ratio = (1.0 / this->ir_);
      }

      vec3 unit_direction = unit(ray.direction());
      double cos_theta = fmin(-unit_direction * correctedNormal, 1.0);
      double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

      bool cannot_refract = refraction_ratio * sin_theta > 1.0;
      vec3 direction;

      if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(state))
        direction = reflect(unit_direction, correctedNormal);
      else
        direction = refract(unit_direction, correctedNormal, refraction_ratio);

      ray = Ray(contact, direction);
      return true;
    }

  protected:
    __device__ double reflectance(double cosine, double ref_idx) const
    {
      // Use Schlick's approximation for reflectance.
      auto r0 = (1 - ref_idx) / (1 + ref_idx);
      r0 = r0 * r0;
      return r0 + (1 - r0) * pow((1 - cosine), 5);
    }

    __device__ vec3 refract(vec3 uv, vec3 &n, double etai_etat) const
    {
      double cos_theta = fmin(-uv * n, 1.0);
      vec3 r_out_perp = etai_etat * (uv + cos_theta * n);
      vec3 r_out_parallel = -sqrt(fabs(1.0 - length_squared(r_out_perp))) * n;
      return r_out_perp + r_out_parallel;
    }

    double ir_;
  };

} // namespace RTW