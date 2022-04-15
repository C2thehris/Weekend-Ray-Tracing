#pragma once
#include <array>
#include <cmath>

template <class T>
class Vec3
{
private:
  std::array<T, 3> e;

public:
  constexpr Vec3() : e{0, 0, 0} {}
  constexpr Vec3(T e0, T e1, T e2) : e{e0, e1, e2} {}

  constexpr T operator[](const int index) const { return e[index]; }

  constexpr T &operator[](const int index) { return e[index]; }

  constexpr T x() const noexcept { return e[0]; }
  constexpr T y() const noexcept { return e[1]; }
  constexpr T z() const noexcept { return e[2]; }

  constexpr Vec3<T> operator-() const noexcept
  {
    return Vec3<T>(-this->e[0], -this->e[1], -this->e[2]);
  }

  constexpr Vec3<T> operator-(const Vec3<T> &rhs) const noexcept
  {
    T x = this->e[0] - rhs.e[0];
    T y = this->e[1] - rhs.e[1];
    T z = this->e[2] - rhs.e[2];
    return Vec3<T>(x, y, z);
  }

  constexpr Vec3<T> operator+(const Vec3<T> &rhs) const noexcept
  {
    T x = this->e[0] + rhs.e[0];
    T y = this->e[1] + rhs.e[1];
    T z = this->e[2] + rhs.e[2];
    return Vec3<T>(x, y, z);
  }

  constexpr T operator*(const Vec3<T> &rhs) const noexcept // Dot Product
  {
    T total = 0;
    total += this->e[0] * rhs.e[0];
    total += this->e[1] * rhs.e[1];
    total += this->e[2] * rhs.e[2];
    return total;
  }

  constexpr Vec3<T> operator*(const T scalar) const noexcept // Scalar Product
  {
    T x = this->x() * scalar;
    T y = this->y() * scalar;
    T z = this->z() * scalar;
    return Vec3<T>(x, y, z);
  }

  constexpr Vec3<T> operator/(const T scalar) const // Scalar Division
  {
    return *this * (1 / scalar);
  }

  constexpr Vec3<T> &operator-=(const Vec3<T> &rhs) noexcept
  {
    this->e[0] -= rhs.e[0];
    this->e[1] -= rhs.e[1];
    this->e[2] -= rhs.e[2];
    return *this;
  }

  constexpr Vec3<T> &operator+=(const Vec3<T> &rhs) noexcept
  {
    this->e[0] += rhs.e[0];
    this->e[1] += rhs.e[1];
    this->e[2] += rhs.e[2];
    return *this;
  }

  constexpr Vec3<T> &operator*=(const T scalar) noexcept // Scalar Product
  {
    for (auto &coord : this->e)
    {
      coord *= scalar;
    }
    return this;
  }

  constexpr Vec3<T> &operator/=(const T scalar) // Scalar Division
  {
    return this *= (1 / scalar);
  }

  constexpr T length_squared() const noexcept
  {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

  constexpr T length() const noexcept
  {
    return std::sqrt(this->length_squared());
  }

  constexpr Vec3<T> unit() const { return *this / this->length(); }

  constexpr bool nearZero() const
  {
    const double eps = .0000001;
    return e[0] < eps && e[1] < eps && e[2] < eps;
  }
};

template <class T>
inline std::ostream &operator<<(std::ostream &ostr,
                                const Vec3<T> &vec) noexcept
{
  ostr << vec.x() << ' ' << vec.y() << ' ' << vec.z();
  return ostr;
}

template <class T>
constexpr Vec3<T> crossProuduct(const Vec3<T> &lhs, const Vec3<T> &rhs) noexcept
{
  T x = lhs.y() * rhs.z() - lhs.z() * rhs.y();
  T y = lhs.z() * rhs.x() - lhs.x() * rhs.z();
  T z = lhs.x() * rhs.y() - lhs.y() * rhs.x();
  return Vec3<T>(x, y, z);
}

template <class T>
constexpr Vec3<T> operator*(const T scalar, const Vec3<T> &vec)
{
  return vec * scalar;
}

template <class T>
constexpr Vec3<T> elementMult(const Vec3<T> &lhs, const Vec3<T> &rhs)
{
  T x = lhs.x() * rhs.x();
  T y = lhs.y() * rhs.y();
  T z = lhs.z() * rhs.z();
  return Vec3<T>(x, y, z);
}

template <class T>
using Point3 = Vec3<T>;

template <class T>
using Color = Vec3<T>;
