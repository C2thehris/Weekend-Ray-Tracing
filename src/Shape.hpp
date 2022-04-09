#include "color.hpp"
#include "ray.hpp"
#include "vec3.hpp"

template <class T>
class Shape {
 public:
  Shape(const Point3<T>& center, const Color<T>& color)
      : center_(center), color_(color) {}

  constexpr virtual bool collision(const Ray<T>& r) const noexcept = 0;
  constexpr const Point3<T> center() const noexcept { return this->center_; }
  constexpr const Color<T> color() const noexcept { return this->color_; }

 private:
  Point3<T> center_;
  Color<T> color_;
};