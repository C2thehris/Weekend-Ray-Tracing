#pragma once
#include <memory>
#include <optional>
#include <vector>

#include "Shape.hpp"
#include "ray.hpp"
#include "rtweekend.hpp"
#include "vec3.hpp"

using std::make_unique;
using std::unique_ptr;

struct Collision_t
{
  double t = INF;
  Shape<double> *contacted;
  Vec3<double> normal;
  Point3<double> contact;
  bool flipped = false;
};

class ShapeContainer
{
private:
  std::vector<unique_ptr<Shape<double>>> shapes;

public:
  void push_back(unique_ptr<Shape<double>> &&s)
  {
    shapes.emplace_back(move(s));
  }

  std::optional<Collision_t> collision(const Ray<double> &ray) const noexcept
  {
    bool hit = false;

    Collision_t soonest;
    for (const auto &ptr : shapes)
    {
      Vec3<double> normal;
      double t = ptr->collision(ray, normal);
      if (t >= 0.001 && t < soonest.t)
      {
        hit = true;
        soonest.t = t;
        soonest.normal = normal;
        soonest.contact = ray.at(t);
        soonest.contacted = ptr.get();
      }
    }
    if (!hit)
      return std::nullopt;
    return soonest;
  }
};