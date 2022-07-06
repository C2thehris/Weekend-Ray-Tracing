#pragma once

#include <iostream>
#include <fstream>
#include <exception>
#include <chrono>
#include <ctime>
#include <string>

#include "rtweekend_cuda.cuh"
#include "camera.cuh"
#include "shape_container.cuh"

namespace RTW
{
  const int nFaces = 1;
  const int nSpheres = 4;

  __global__ void allocateShapes(Shape **shapes)
  {
    Material *lambertianGround = new Lambertian(make_double3(.8, .8, 0));
    Material *lambertianCenter = new Lambertian(make_double3(.1, .2, .5));
    Material *dielecticLeft = new Dielectic(make_double3(1, 1, 1), 1.5);
    Material *metalRight = new Metal(make_double3(.8, .6, .2), 1.0);

    shapes[0] = new Face(make_double3(0.0, -1.0, 0), lambertianGround, make_double3(0, -1, 0));
    shapes[1] = new Sphere(make_double3(0, 0, -1), lambertianCenter, .5);
    shapes[2] = new Sphere(make_double3(-1, 0, -1), dielecticLeft, .5);
    shapes[3] = new Sphere(make_double3(-1, 0, -1), dielecticLeft, -.45);
    shapes[4] = new Sphere(make_double3(1, 0, -1), metalRight, .5);
  }

  __global__ void allocateCamera(Camera **camera, double aspect_ratio, dim3 imgDim)
  {
    point3 lookFrom = make_double3(-2, 2, 1);
    point3 lookTo = make_double3(0, 0, -1);
    vec3 vup = make_double3(0, 1, 0);
    *camera = new Camera(lookFrom, lookTo, vup, 20, aspect_ratio);
  }

  __global__ void allocateContainer(ShapeContainer **container, Shape **shapes, int nShapes)
  {
    *container = new ShapeContainer(shapes, nShapes);
  }

  __device__ color3 rayColor(Ray r, ShapeContainer *shapes, curandState_t *state)
  {
    double3 colors[DEPTH_LIMIT];

    int i = 0;
    while (true)
    {
      Collision_t closest;
      if (shapes->collision(r, closest, state))
      {
        bool propogate = closest.mat->scatter(r, closest.contact, closest.normal, state);
        if (i + 1 == DEPTH_LIMIT || !propogate)
        {
          colors[i] = double3();
          break;
        }
        colors[i] = closest.mat->color();
        i += 1;
        continue;
      }

      double3 unit_vec = unit(r.direction());
      double t = .5 * (unit_vec.y + 1.0);

      colors[i] = (1.0 - t) * make_double3(1.0, 1.0, 1.0) + t * make_double3(.5, .7, 1.0);
      break;
    }

    for (int j = i - 1; j >= 0; j -= 1)
    {
      colors[j] = elementMult(colors[j], colors[j + 1]);
    }

    return colors[0];
  }

  __global__ void renderPixel(color3 *img, dim3 imgDim, Camera **camera, ShapeContainer **container, int seed)
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= imgDim.x)
      return;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= imgDim.y)
      return;

    int index = x + (imgDim.y - y - 1) * imgDim.x;

    curandState_t state;
    curand_init(seed, index << 16, 0, &state);

    for (int i = 0; i < SAMPLES; i += 1)
    {
      double u = static_cast<double>(x + random_double(&state)) / (imgDim.x - 1);
      double v = static_cast<double>(y + random_double(&state)) / (imgDim.y - 1);
      img[index] += rayColor((*camera)->getRay(u, v), *container, &state);
    }
  }

  class Renderer
  {
  public:
    __host__ static color3 *render(dim3 imgDim, std::string imageLayoutFile)
    {
      int pixelCount = imgDim.x * imgDim.y;
      double aspect_ratio = static_cast<double>(imgDim.x) / static_cast<double>(imgDim.y);

      Camera **camera;
      hostCheckError(cudaMalloc(&camera, sizeof(Camera *)));
      allocateCamera<<<1, 1>>>(camera, aspect_ratio, imgDim);

      color3 *devImage;
      hostCheckError(cudaMalloc(&devImage, sizeof(color3) * pixelCount));

      Shape **shapes;
      hostCheckError(cudaMalloc(&shapes, sizeof(Shape *) * 5));
      allocateShapes<<<1, 1>>>(shapes);

      ShapeContainer **container;
      hostCheckError(cudaMalloc(&container, sizeof(ShapeContainer **)));
      allocateContainer<<<1, 1>>>(container, shapes, 5);

      dim3 threads(NUM_THREADS, NUM_THREADS);
      dim3 blocks((imgDim.x / NUM_THREADS) + 1, (imgDim.y / NUM_THREADS) + 1);

      srand(std::time(0));

      hostCheckError(cudaDeviceSynchronize());

      renderPixel<<<blocks, threads>>>(devImage, imgDim, camera, container, rand());
      hostCheckError(cudaDeviceSynchronize());

      // TODO: Free Stuff

      color3 *hostImage;
      hostCheckError(cudaMallocHost(&hostImage, sizeof(color3) * pixelCount));

      hostCheckError(cudaMemcpy(hostImage, devImage, sizeof(color3) * pixelCount, cudaMemcpyDeviceToHost));
      return hostImage;
    }

  private:
    Renderer() {}

    static Renderer instance;
  };
}; // class Renderer
