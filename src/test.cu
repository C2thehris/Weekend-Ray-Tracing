#include <iostream>
#include <fstream>
#include <exception>
#include <chrono>
#include <ctime>

#include "rtweekend_cuda.cuh"
#include "camera.cuh"
#include "shape_container.cuh"

#define NUM_THREADS 16

__device__ color3 rayColor(const Ray &r, ShapeContainer *shapes, curandState_t *state)
{
  Collision_t closest;
  if (shapes->collision(r, closest))
  {
    return closest.color;
  }
  double3 unit_vec = unit(r.direction());
  double t = .5 * (unit_vec.y + 1.0);
  return (1.0 - t) * make_double3(1.0, 1.0, 1.0) + t * make_double3(.5, .7, 1.0);
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
  img[index] /= SAMPLES;
}

ShapeContainer *allocateContainer()
{
  const int n = 1;
  Sphere redSphere(make_double3(0, 0, -1), make_double3(1, 0, 0), .5);
  Sphere *spheres;
  hostCheckError(cudaMallocHost(&spheres, n * sizeof(Sphere)));
  spheres[0] = redSphere;

  Sphere *dev_spheres;
  hostCheckError(cudaMalloc(&dev_spheres, n * sizeof(Sphere)));
  hostCheckError(cudaMemcpy(dev_spheres, spheres, n * sizeof(Sphere), cudaMemcpyHostToDevice));

  Face *faces;
  hostCheckError(cudaMallocHost(&faces, n * sizeof(Face)));
  Face greenFace(make_double3(0.0, -1.0, 0), make_double3(.8, .8, 0), make_double3(0, -1, 0));
  faces[0] = greenFace;

  Face *dev_faces;
  hostCheckError(cudaMalloc(&dev_faces, n * sizeof(Face)));
  hostCheckError(cudaMemcpy(dev_faces, faces, n * sizeof(Face), cudaMemcpyHostToDevice));

  hostCheckError(cudaFreeHost(spheres));
  hostCheckError(cudaFreeHost(faces));

  ShapeContainer shapes(dev_spheres, 1, dev_faces, 1);
  ShapeContainer *dev_container;
  hostCheckError(cudaMalloc(&dev_container, sizeof(ShapeContainer)));
  hostCheckError(cudaMemcpy(dev_container, &shapes, sizeof(ShapeContainer), cudaMemcpyHostToDevice));
  return dev_container;
}

Camera *allocateCamera(double aspect_ratio, dim3 imgDim)
{
  point3 lookFrom = make_double3(0, 0, 0);
  point3 lookTo = make_double3(0, 0, -1);
  vec3 vup = make_double3(0, 1, 0);
  Camera view(lookFrom, lookTo, vup, 105, aspect_ratio);
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
  ShapeContainer *device_container = allocateContainer();

  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 blocks((imgDim.x / NUM_THREADS) + 1, (imgDim.y / NUM_THREADS) + 1);

  auto t_start = std::chrono::high_resolution_clock::now();

  srand(std::time(0));
  renderPixel<<<blocks, threads>>>(img_segment, imgDim, camera, device_container, rand());

  hostCheckError(cudaDeviceSynchronize());

  auto t_end = std::chrono::high_resolution_clock::now();

  std::cout << "Rendering took " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms \n";

  color3 *host_img;
  hostCheckError(cudaMallocHost(&host_img, pixelCount * sizeof(color3)));
  hostCheckError(cudaMemcpy(host_img, img_segment, pixelCount * sizeof(color3), cudaMemcpyDeviceToHost));

  hostCheckError(cudaFree(img_segment));
  hostCheckError(cudaFree(camera));
  hostCheckError(cudaFree(device_container));

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
