#include <unistd.h>

#include <iostream>

#include "Sphere.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "vec3.hpp"

Color<double> ray_color(const Ray<double> &ray, const Shape<double> &shape) {
  if (shape.collision(ray)) return shape.color();
  Vec3<double> unit = ray.direction().unit();
  double t = .5 * (unit.y() + 1);
  return (1.0 - t) * Color<double>(1, 1, 1) + t * Color<double>(.5, .7, 1.0);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Invalid Usage\n";
    exit(1);
  }
  const int image_width = atoi(argv[1]);
  const int image_height = atoi(argv[2]);
  const double aspect_ratio = static_cast<double>(image_width) / image_height;

  const double viewport_height = 2.0;
  const double viewport_width = viewport_height * aspect_ratio;
  const double focal_length = 1.0;

  auto origin = Point3<double>(0, 0, 0);
  auto horizontal = Vec3<double>(viewport_width, 0, 0);
  auto vertical = Vec3<double>(0, viewport_height, 0);
  auto lower_left =
      origin - horizontal / 2 - vertical / 2 - Vec3<double>(0, 0, focal_length);
  Sphere<double> center_sphere(Vec3<double>(0, 0, -1), Color<double>(1, 0, 0),
                               .5);

  std::cout << "P3" << std::endl
            << image_width << ' ' << image_height << std::endl
            << "255" << std::endl;
  for (int i = image_height - 1; i >= 0; i -= 1) {
    std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
    for (int j = 0; j < image_width; j += 1) {
      double u = double(j) / (image_width - 1);
      double v = double(i) / (image_height - 1);
      Ray<double> r(origin,
                    lower_left + u * horizontal + v * vertical - origin);
      Color<double> pixel_color = ray_color(r, center_sphere);

      std::cout << write_color(pixel_color) << '\n';
    }
  }
  std::cerr << std::endl;

  return EXIT_SUCCESS;
}