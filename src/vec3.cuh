#pragma once

#ifndef __CUDA_ARCH__
#include <cmath>
#endif

__host__ __device__ double3 operator-(const double3 &vec) noexcept
{
  return make_double3(-vec.x, -vec.y, -vec.z);
}

__host__ __device__ double3 operator-(const double3 &lhs, const double3 &rhs) noexcept
{
  double x = lhs.x - rhs.x;
  double y = lhs.y - rhs.y;
  double z = lhs.z - rhs.z;
  return make_double3(x, y, z);
}

__host__ __device__ double3 operator+(const double3 &lhs, const double3 &rhs) noexcept
{
  double x = lhs.x + rhs.x;
  double y = lhs.y + rhs.y;
  double z = lhs.z + rhs.z;
  return make_double3(x, y, z);
}

__host__ __device__ double operator*(const double3 &lhs, const double3 &rhs) noexcept // Dot Product
{
  double total = 0;
  total += lhs.x * rhs.x;
  total += lhs.y * rhs.y;
  total += lhs.z * rhs.z;
  return total;
}

__host__ __device__ double3 operator*(const double3 &lhs, const double scalar) noexcept // Scalar Product
{
  double x = lhs.x * scalar;
  double y = lhs.y * scalar;
  double z = lhs.z * scalar;
  return make_double3(x, y, z);
}

__host__ __device__ double3 operator/(const double3 &lhs, const double scalar) // Scalar Division
{
  return lhs * (1 / scalar);
}

__host__ __device__ double3 &operator-=(double3 &lhs, const double3 &rhs) noexcept
{
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  lhs.z -= rhs.z;
  return lhs;
}

__host__ __device__ double3 &operator+=(double3 &lhs, const double3 &rhs) noexcept
{
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  return lhs;
}

__host__ __device__ double3 &operator*=(double3 &lhs, const double scalar) noexcept // Scalar Product
{
  lhs.x *= scalar;
  lhs.y *= scalar;
  lhs.z *= scalar;
  return lhs;
}

__host__ __device__ double3 &operator/=(double3 &lhs, const double scalar) // Scalar Division
{
  return lhs *= (1 / scalar);
}

__host__ __device__ double length_squared(const double3 &vec) noexcept
{
  return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

__host__ __device__ double length(const double3 &vec) noexcept
{
#ifdef __CUDA_ARCH__
  return norm3d(vec.x, vec.y, vec.z);
#else
  return std::sqrt(length_squared(vec));
#endif
}

__host__ __device__ double3 unit(double3 vec)
{
  return vec / length(vec);
}

__host__ __device__ bool nearZero(double3 vec)
{
  const double eps = .0000001;
  return vec.x < eps && vec.y < eps && vec.z < eps;
}

__host__ __device__ double3 crossProuduct(const double3 &lhs, const double3 &rhs) noexcept
{
  double x = lhs.y * rhs.z - lhs.z * rhs.y;
  double y = lhs.z * rhs.x - lhs.x * rhs.z;
  double z = lhs.x * rhs.y - lhs.y * rhs.x;
  return make_double3(x, y, z);
}

__host__ __device__ double3 operator*(const double scalar, const double3 &vec)
{
  return vec * scalar;
}

__host__ __device__ double3 elementMult(const double3 &lhs, const double3 &rhs)
{
  double x = lhs.x * rhs.x;
  double y = lhs.y * rhs.y;
  double z = lhs.z * rhs.z;
  return make_double3(x, y, z);
}
