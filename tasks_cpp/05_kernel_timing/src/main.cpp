#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char *kKernelSource =
    "__kernel void scale(__global float* x, float alpha) {"
    "  const int gid = get_global_id(0);"
    "  x[gid] *= alpha;"
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
        const size_t n = 1 << 20;
        std::vector<float> data(n, 1.0f);

        const cl::Device device = pick_first_device();
        cl::Context context(device);
        const cl_queue_properties queue_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cl::CommandQueue queue(context, device, queue_props);
        cl::Program program(context, kKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "scale");

        cl::Buffer data_buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, data.data());
        kernel.setArg(0, data_buf);
        kernel.setArg(1, 2.0f);

        cl::Event evt;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange, nullptr, &evt);
        evt.wait();

        const cl_ulong t0 = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        const cl_ulong t1 = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        const double kernel_ms = static_cast<double>(t1 - t0) / 1e6;

        std::cout << "Kernel execution time: " << kernel_ms << " ms\n";
        return EXIT_SUCCESS;
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
