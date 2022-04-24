#include <iostream>
#include <fstream>
#include <exception>
#include <chrono>
#include <ctime>

#include "rtweekend_cuda.cuh"
#include "camera.cuh"
#include "shape_container.cuh"

#define NUM_THREADS 16

__device__ color3 rayColor(Ray r, ShapeContainer *shapes, curandState_t *state)
{
  double3 colors[DEPTH_LIMIT];

  int i = 0;
  while (true)
  {
    Collision_t closest;
    if (shapes->collision(r, closest, state))
    {
      if (i + 1 == DEPTH_LIMIT || !closest.propogate)
      {
        colors[i] = double3();
        break;
      }
      colors[i] = closest.color;
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

__global__ void renderPixel(color3 *img, dim3 imgDim, const Camera *camera, ShapeContainer *container, int seed)
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
    img[index] += rayColor(camera->getRay(u, v), container, &state);
  }
}

Face *allocateFaces(int &n_faces)
{
  n_faces = 1;
  Face *faces;
  hostCheckError(cudaMallocHost(&faces, n_faces * sizeof(Face)));
  MaterialProperty lambertianGround;
  lambertianGround.type = LAMBERTIAN;
  Face greenFace(make_double3(0.0, -1.0, 0), lambertianGround, make_double3(.8, .8, 0), make_double3(0, -1, 0));
  faces[0] = greenFace;

  Face *dev_faces;
  hostCheckError(cudaMalloc(&dev_faces, n_faces * sizeof(Face)));
  hostCheckError(cudaMemcpy(dev_faces, faces, n_faces * sizeof(Face), cudaMemcpyHostToDevice));
  hostCheckError(cudaFreeHost(faces));
  return dev_faces;
}

Sphere *allocateSpheres(int &n_spheres)
{
  n_spheres = 4;
  Sphere *spheres;
  hostCheckError(cudaMallocHost(&spheres, n_spheres * sizeof(Sphere)));

  MaterialProperty lambertianCenter;
  lambertianCenter.type = LAMBERTIAN;
  MaterialProperty dielecticLeft;
  dielecticLeft.type = DIELECTIC;
  dielecticLeft.property = 1.5;
  MaterialProperty metalRight;
  metalRight.type = METAL;
  metalRight.property = 1.0;

  Sphere centerSphere(make_double3(0, 0, -1), lambertianCenter, make_double3(.1, .2, .5), 1.0);
  Sphere leftSphere(make_double3(-1.5, 0, -1), dielecticLeft, make_double3(.8, .8, .8), .5);
  Sphere leftSmall(make_double3(-1.5, 0, -1), dielecticLeft, make_double3(.8, .8, .8), -.4);
  Sphere rightSphere(make_double3(2.0, 0, -1), metalRight, make_double3(.8, .6, .2), 1.0);
  spheres[0] = centerSphere;
  spheres[1] = leftSphere;
  spheres[2] = leftSmall;
  spheres[3] = rightSphere;

  Sphere *dev_spheres;
  hostCheckError(cudaMalloc(&dev_spheres, n_spheres * sizeof(Sphere)));
  hostCheckError(cudaMemcpy(dev_spheres, spheres, n_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));
  hostCheckError(cudaFreeHost(spheres));
  return dev_spheres;
}

Camera *allocateCamera(double aspect_ratio, dim3 imgDim)
{
  point3 lookFrom = make_double3(-2, 2, 1);
  point3 lookTo = make_double3(0, 0, -1);
  vec3 vup = make_double3(0, 1, 0);
  Camera view(lookFrom, lookTo, vup, 135, aspect_ratio);
  Camera *camera;

  hostCheckError(cudaMalloc(&camera, sizeof(Camera)));
  hostCheckError(cudaMemcpy(camera, &view, sizeof(Camera), cudaMemcpyHostToDevice));
  return camera;
}

color3 *allocateImage(int pixel_count)
{
  color3 *img_segment;
  hostCheckError(cudaMalloc(&img_segment, pixel_count * sizeof(color3)));
  return img_segment;
}

int main(int argc, char **argv)
{
  dim3 imgDim;
  double aspect_ratio;
  std::ofstream ostr(argv[3]);
  try
  {
    imgDim.x = std::stoi(argv[1]);
    imgDim.y = std::stoi(argv[2]);
    aspect_ratio = static_cast<double>(imgDim.x) / static_cast<double>(imgDim.y);
  }
  catch (const std::exception &e)
  {
    std::cerr << "INVALID USAGE" << std::endl;
    exit(EXIT_FAILURE);
  }

  int pixelCount = imgDim.x * imgDim.y;

  Camera *camera = allocateCamera(aspect_ratio, imgDim);
  color3 *img_segment = allocateImage(pixelCount);

  int n_spheres, n_faces;
  Sphere *dev_spheres = allocateSpheres(n_spheres);
  Face *dev_faces = allocateFaces(n_faces);

  ShapeContainer shapes(dev_spheres, n_spheres, dev_faces, n_faces);
  ShapeContainer *dev_container;
  hostCheckError(cudaMalloc(&dev_container, sizeof(ShapeContainer)));
  hostCheckError(cudaMemcpy(dev_container, &shapes, sizeof(ShapeContainer), cudaMemcpyHostToDevice));

  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 blocks((imgDim.x / NUM_THREADS) + 1, (imgDim.y / NUM_THREADS) + 1);

  srand(std::time(0));

  auto t_start = std::chrono::high_resolution_clock::now();
  renderPixel<<<blocks, threads>>>(img_segment, imgDim, camera, dev_container, rand());
  hostCheckError(cudaDeviceSynchronize());
  auto t_end = std::chrono::high_resolution_clock::now();

  std::cout << "Rendering took " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms \n";

  color3 *host_img;
  hostCheckError(cudaMallocHost(&host_img, pixelCount * sizeof(color3)));
  hostCheckError(cudaMemcpy(host_img, img_segment, pixelCount * sizeof(color3), cudaMemcpyDeviceToHost));

  hostCheckError(cudaFree(img_segment));
  hostCheckError(cudaFree(camera));
  hostCheckError(cudaFree(dev_container));

  ostr << "P3" << '\n'
       << imgDim.x << ' ' << imgDim.y << '\n'
       << "255" << '\n';
  int i = 0;
  for (; i < pixelCount; i += 1)
  {
    ostr << host_img[i] << '\n';
  }
  ostr.close();
  hostCheckError(cudaFreeHost(host_img));
  std::cout << "Pixels Rendered: " << i << std::endl;
}
