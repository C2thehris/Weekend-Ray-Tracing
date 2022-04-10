
#include "rtweekend.hpp"
#include "Material.hpp"

class Metal : public Material<double>
{
  Metal(const Color<double> &color) : Material<double>(color) {}
};