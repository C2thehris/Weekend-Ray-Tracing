#include <unistd.h>

#include <iostream>
#include <memory>
#include <vector>

#include "Camera.hpp"
#include "Face.hpp"
#include "ShapeContainer.hpp"
#include "Sphere.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include "Material.hpp"
#include "Lambertian.hpp"
#include "Metal.hpp"
#include "Dielectic.hpp"

using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
using std::weak_ptr;

Color<double> ray_color(const Ray<double> &ray, const ShapeContainer &shapes, int depth)
{
  if (depth > DEPTH_LIMIT)
  {
    return Color<double>();
  }

  std::optional<Collision_t> collision = shapes.collision(ray);
  if (collision)
  {
    Color<double> color;
    Ray<double> next;
    // if (depth == 1 && collision.value().contacted->material()->attenuation().z() == .9)
    // {
    //   std::cerr << "HI" << std::endl;
    // }

    if (collision.value().contacted->material()->scatter(ray, collision->contact, collision->normal, next, color))
    {
      return elementMult(color, ray_color(next, shapes, depth + 1));
    }
    return Color<double>();
  }
  Vec3<double> unit = ray.direction().unit();
  double t = .5 * (unit.y() + 1);
  return (1.0 - t) * Color<double>(1, 1, 1) + t * Color<double>(.5, .7, 1.0);
}

void render(int image_width, int image_height, const Camera &view,
            const ShapeContainer &shapes)
{
  std::cout << "P3" << std::endl
            << image_width << ' ' << image_height << std::endl
            << "255" << std::endl;
  for (int i = image_height - 1; i >= 0; i -= 1)
  {
    std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
    for (int j = 0; j < image_width; j += 1)
    {
      Color<double> pixel_color;
      for (int s = 0; s < SAMPLES; s += 1)
      {
        double u = double(j + random_double()) / (image_width - 1);
        double v = double(i + random_double()) / (image_height - 1);
        pixel_color += ray_color(view.getRay(u, v), shapes, 0);
      }

      std::string out = write_color(pixel_color);
      std::cout << write_color(pixel_color) << '\n';
    }
  }
  std::cerr << std::endl;
}

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cerr << "Invalid Usage\n";
    exit(1);
  }
  const int image_width = atoi(argv[1]);
  const int image_height = atoi(argv[2]);
  const double aspect_ratio = static_cast<double>(image_width) / static_cast<double>(image_height);

  Point3<double> origin(-2, 2, 1);
  Point3<double> target(0, 0, -1);
  Vec3<double> vup(0, 1, 0);
  Camera view(origin, target, vup, 135.0, aspect_ratio);

  unique_ptr<Material<double>> material_ground = make_unique<Lambertian>(Color<double>(0.8, 0.8, 0.0));
  unique_ptr<Material<double>> material_center = make_unique<Lambertian>(Color<double>(0.1, 0.2, 0.5));
  unique_ptr<Material<double>> material_left = make_unique<Dielectic>(1.5);
  unique_ptr<Material<double>> material_left2 = make_unique<Dielectic>(1.5);
  unique_ptr<Material<double>> material_right = make_unique<Metal>(Color<double>(0.8, 0.6, 0.2), 0.0);

  ShapeContainer shapes;
  shapes.push_back(move(make_unique<Face<double>>(Point3<double>(0.0, -1.0, 0), move(material_ground), Vec3<double>(0.0, -1.0, 0.0))));
  // shapes.push_back(move(make_unique<Sphere<double>>(Point3<double>(0.0, -100.5, 0), move(material_ground), 100)));
  shapes.push_back(move(make_unique<Sphere<double>>(Point3<double>(0.0, 0.0, -1.0), move(material_center), 1.0)));
  shapes.push_back(move(make_unique<Sphere<double>>(Point3<double>(-1.5, 0.0, -1.0), move(material_left), .5)));
  shapes.push_back(move(make_unique<Sphere<double>>(Point3<double>(-1.5, 0.0, -1.0), move(material_left2), -.4)));
  shapes.push_back(move(make_unique<Sphere<double>>(Point3<double>(2.0, 0.0, -1.0), move(material_right), 1.0)));

  render(image_width, image_height, view, shapes);

  return EXIT_SUCCESS;
}
