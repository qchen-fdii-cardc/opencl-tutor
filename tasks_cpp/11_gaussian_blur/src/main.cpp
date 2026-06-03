#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char *kKernelSource =
    "__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;"
    "__kernel void gaussian3x3(read_only image2d_t src, write_only image2d_t dst, int w, int h) {"
    "  int x = get_global_id(0);"
    "  int y = get_global_id(1);"
    "  const float k[3][3] = {{1.0f,2.0f,1.0f},{2.0f,4.0f,2.0f},{1.0f,2.0f,1.0f}};"
    "  float4 sum = (float4)(0.0f);"
    "  float weight = 0.0f;"
    "  for (int j=-1; j<=1; ++j) for (int i=-1; i<=1; ++i) {"
    "    int2 p = (int2)(clamp(x+i,0,w-1), clamp(y+j,0,h-1));"
    "    float wv = k[j+1][i+1];"
    "    sum += read_imagef(src, smp, p) * wv;"
    "    weight += wv;"
    "  }"
    "  write_imagef(dst, (int2)(x,y), sum / weight);"
    "}";

static cl::Device pick_first_device()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (size_t i = 0; i < platforms.size(); ++i)
    {
        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (!devices.empty())
            return devices[0];
    }
    throw std::runtime_error("No usable OpenCL device found.");
}

int main()
{
    try
    {
        const int w = 32;
        const int h = 32;
        const size_t pixels = static_cast<size_t>(w * h);

        std::vector<cl_float4> src(pixels), dst(pixels);
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                cl_float4 p;
                p.s[0] = ((x / 8 + y / 8) % 2) ? 1.0f : 0.0f;
                p.s[1] = p.s[0];
                p.s[2] = p.s[0];
                p.s[3] = 1.0f;
                src[static_cast<size_t>(y * w + x)] = p;
            }
        }

        const cl::Device device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, kKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "gaussian3x3");

        cl::ImageFormat fmt(CL_RGBA, CL_FLOAT);
        cl::Image2D src_img(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fmt, w, h, 0, src.data());
        cl::Image2D dst_img(context, CL_MEM_WRITE_ONLY, fmt, w, h);

        kernel.setArg(0, src_img);
        kernel.setArg(1, dst_img);
        kernel.setArg(2, w);
        kernel.setArg(3, h);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h), cl::NullRange);

        cl::size_t<3> origin;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        cl::size_t<3> region;
        region[0] = static_cast<size_t>(w);
        region[1] = static_cast<size_t>(h);
        region[2] = 1;
        queue.enqueueReadImage(dst_img, CL_TRUE, origin, region, 0, 0, dst.data());

        bool changed = false;
        for (size_t i = 0; i < pixels; ++i)
        {
            if (std::fabs(dst[i].s[0] - src[i].s[0]) > 1e-3f)
            {
                changed = true;
                break;
            }
        }

        std::cout << (changed ? "Gaussian blur task verified.\n" : "Gaussian blur task failed.\n");
        return changed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const cl::Error &e)
    {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
