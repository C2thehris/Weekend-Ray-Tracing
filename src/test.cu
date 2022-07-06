#include "renderer.cuh"

int main(int argc, char **argv)
{
  dim3 imgDim;
  std::ofstream ostr(argv[3]);
  try
  {
    imgDim.x = std::stoi(argv[1]);
    imgDim.y = std::stoi(argv[2]);
    if (imgDim.y == 0)
    {
      throw "";
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "INVALID USAGE" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  color3 *image = RTW::Renderer::render(imgDim, "input.txt");

  auto t_end = std::chrono::high_resolution_clock::now();

  std::cout << "Rendering took " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms \n";

  ostr << "P3" << '\n'
       << imgDim.x << ' ' << imgDim.y << '\n'
       << "255" << '\n';
  int i = 0;
  int pixelCount = imgDim.x * imgDim.y;
  for (; i < pixelCount; i += 1)
  {
    ostr << image[i] << '\n';
  }
  ostr.close();
  hostCheckError(cudaFreeHost(image));
  std::cout << "Pixels Rendered: " << i << std::endl;
}
