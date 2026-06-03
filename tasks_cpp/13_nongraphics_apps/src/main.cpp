#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char* kKernelSource =
    "__kernel void monte_carlo_hits(__global const uint* seeds, __global uint* hits, int samples_per_item) {"
    "  const int gid = get_global_id(0);"
    "  uint state = seeds[gid];"
    "  uint count = 0;"
    "  for (int i = 0; i < samples_per_item; ++i) {"
    "    state = state * 1664525u + 1013904223u;"
    "    float x = (float)(state & 0x00FFFFFF) / 16777216.0f;"
    "    state = state * 1664525u + 1013904223u;"
    "    float y = (float)(state & 0x00FFFFFF) / 16777216.0f;"
    "    if (x*x + y*y <= 1.0f) ++count;"
    "  }"
    "  hits[gid] = count;"
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
        const size_t work_items = 4096;
        const int samples_per_item = 1024;
        std::vector<cl_uint> seeds(work_items), hits(work_items, 0);
        for (size_t i = 0; i < work_items; ++i) {
            seeds[i] = static_cast<cl_uint>(1234567u + i * 2654435761u);
        }

        const cl::Device device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, kKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "monte_carlo_hits");

        cl::Buffer seed_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * work_items, seeds.data());
        cl::Buffer hit_buf(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * work_items);

        kernel.setArg(0, seed_buf);
        kernel.setArg(1, hit_buf);
        kernel.setArg(2, samples_per_item);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(work_items), cl::NullRange);
        queue.enqueueReadBuffer(hit_buf, CL_TRUE, 0, sizeof(cl_uint) * work_items, hits.data());

        unsigned long long total_hits = 0;
        for (size_t i = 0; i < work_items; ++i) total_hits += hits[i];
        const double total_samples = static_cast<double>(work_items) * samples_per_item;
        const double pi_est = 4.0 * static_cast<double>(total_hits) / total_samples;

        std::cout << "Estimated PI: " << pi_est << "\n";
        const bool ok = (pi_est > 3.05 && pi_est < 3.25);
        std::cout << (ok ? "Non-graphics OpenCL task verified.\n" : "Non-graphics OpenCL task failed.\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
