#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cmath>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char* kKernelSource =
    "__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;"
    "__kernel void copy_image(read_only image2d_t src, write_only image2d_t dst) {"
    "  int2 pos = (int2)(get_global_id(0), get_global_id(1));"
    "  float4 p = read_imagef(src, smp, pos);"
    "  write_imagef(dst, pos, p);"
    "}";

static cl::Device pick_first_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (size_t i = 0; i < platforms.size(); ++i) {
        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (!devices.empty()) return devices[0];
    }
    throw std::runtime_error("No usable OpenCL device found.");
}

int main() {
    try {
        const int w = 32;
        const int h = 32;
        const size_t pixels = static_cast<size_t>(w * h);
        std::vector<cl_float4> src(pixels), dst(pixels);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                cl_float4 p;
                p.s[0] = static_cast<float>(x) / w;
                p.s[1] = static_cast<float>(y) / h;
                p.s[2] = 0.5f;
                p.s[3] = 1.0f;
                src[static_cast<size_t>(y * w + x)] = p;
            }
        }

        const cl::Device device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, kKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "copy_image");

        cl::ImageFormat fmt(CL_RGBA, CL_FLOAT);
        cl::Image2D src_img(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fmt, w, h, 0, src.data());
        cl::Image2D dst_img(context, CL_MEM_WRITE_ONLY, fmt, w, h);

        kernel.setArg(0, src_img);
        kernel.setArg(1, dst_img);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h), cl::NullRange);

        std::array<size_t, 3> origin = {{0, 0, 0}};
        std::array<size_t, 3> region = {{static_cast<size_t>(w), static_cast<size_t>(h), 1}};
        queue.enqueueReadImage(dst_img, CL_TRUE, origin, region, 0, 0, dst.data());

        bool ok = true;
        for (size_t i = 0; i < pixels; ++i) {
            if (std::fabs(dst[i].s[0] - src[i].s[0]) > 1e-6f ||
                std::fabs(dst[i].s[1] - src[i].s[1]) > 1e-6f) {
                ok = false;
                break;
            }
        }

        std::cout << (ok ? "Image and sampler task verified.\n" : "Image and sampler task failed.\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
