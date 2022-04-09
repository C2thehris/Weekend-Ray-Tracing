#include "Shape.hpp"

template <class T>
class Sphere : public Shape<T> {
 public:
  constexpr Sphere(const Vec3<T>& center, const Vec3<T>& color,
                   const T radius) noexcept
      : Shape<T>(center, color), radius_(radius) {}

  constexpr bool collision(const Ray<T>& r) const noexcept override {
    auto RminusC = r.origin() - this->center();
    double a = r.direction() * r.direction();
    double b = 2.0 * r.direction() * (RminusC);
    double c = (RminusC * RminusC) - (radius_ * radius_);
    double discriminant = b * b - 4.0 * a * c;
    return discriminant > 0;
  }

 private:
  T radius_;
};