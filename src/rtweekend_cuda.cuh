#pragma once
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

#include <curand_kernel.h>
#include "vec3.cuh"

#define MAX_COLOR 255.99999
#define SAMPLES 100
#define DEPTH_LIMIT 50
#define PI 3.14159265358979323
#define INF std::numeric_limits<double>::infinity()

typedef double3 point3;
typedef double3 vec3;
typedef double3 color3;

inline void hostCheckError(cudaError_t error)
{
  if (error != cudaSuccess)
  {
    std::cerr << cudaGetErrorString(error) << '\n';
  }
}

// TODO: Use curand to produce a random double
__device__ double random_double(curandState_t *state)
{
  return curand_uniform_double(state);
  // return 0.0;
}

__device__ vec3 random_unit_vector(curandState_t *state)
{
  double x, y, z;
  do
  {
    x = random_double(state) * 2 - 1;
    y = random_double(state) * 2 - 1;
    z = random_double(state) * 2 - 1;
  } while (norm3d(x, y, z) > 1);

  return unit(make_double3(x, y, z));
}

constexpr double clamp(double val, double low, double high)
{
  return std::max(std::min(val, high), low);
}

constexpr double degrees_to_radians(double degrees)
{
  return (degrees / 180.0) * PI;
}

std::ostream &operator<<(std::ostream &ostr,
                         const dim3 &vec) noexcept
{
  ostr << vec.x << ' ' << vec.y << ' ' << vec.z;
  return ostr;
}

std::ostream &operator<<(std::ostream &ostr,
                         const color3 &vec) noexcept
{
  ostr << static_cast<int>(vec.x * MAX_COLOR) << ' ' << static_cast<int>(vec.y * MAX_COLOR) << ' ' << static_cast<int>(vec.z * MAX_COLOR);
  return ostr;
}
