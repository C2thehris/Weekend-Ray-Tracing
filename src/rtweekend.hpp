#pragma once
#include <algorithm>
#include <numeric>
#include <random>

#include "ray.hpp"
#include "vec3.hpp"

#define MAX_COLOR 256
#define SAMPLES 100
#define DEPTH_LIMIT 50
#define INF std::numeric_limits<double>::infinity()

inline double random_double()
{
  // static std::uniform_real_distribution<double> dist(0, 1.0);
  // static std::mt19937 generator;
  // return dist(generator);
  double result = static_cast<double>(rand()) / RAND_MAX;
  return result;
}

inline Vec3<double> random_unit_vector()
{
  double x, y, z;
  Vec3<double> random_vector;
  do
  {
    x = random_double() * 2 - 1;
    y = random_double() * 2 - 1;
    z = random_double() * 2 - 1;
    random_vector = Vec3<double>(x, y, z);
  } while (random_vector.length_squared() > 1);
  return random_vector.unit();
}

inline double clamp(double val, double low, double high)
{
  return std::max(std::min(val, high), low);
}