#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static std::string device_type_to_string(cl_device_type type)
{
    if (type & CL_DEVICE_TYPE_GPU)
        return "GPU";
    if (type & CL_DEVICE_TYPE_CPU)
        return "CPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR)
        return "ACCELERATOR";
    if (type & CL_DEVICE_TYPE_DEFAULT)
        return "DEFAULT";
    if (type & CL_DEVICE_TYPE_CUSTOM)
        return "CUSTOM";
    return "UNKNOWN";
}

static double bytes_to_mib(cl_ulong bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

int main()
{
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            std::cerr << "No OpenCL platform found\n";
            return EXIT_FAILURE;
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "OpenCL platforms found: " << platforms.size() << "\n\n";

        for (size_t pi = 0; pi < platforms.size(); ++pi)
        {
            const cl::Platform &platform = platforms[pi];
            const std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
            const std::string platform_vendor = platform.getInfo<CL_PLATFORM_VENDOR>();
            const std::string platform_version = platform.getInfo<CL_PLATFORM_VERSION>();

            std::cout << "[Platform " << pi << "] " << platform_name << "\n";
            std::cout << "  Vendor: " << platform_vendor << "\n";
            std::cout << "  Version: " << platform_version << "\n";

            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            std::cout << "  Devices: " << devices.size() << "\n";

            for (size_t di = 0; di < devices.size(); ++di)
            {
                const cl::Device &device = devices[di];
                const std::string name = device.getInfo<CL_DEVICE_NAME>();
                const std::string vendor = device.getInfo<CL_DEVICE_VENDOR>();
                const std::string driver = device.getInfo<CL_DRIVER_VERSION>();
                const std::string version = device.getInfo<CL_DEVICE_VERSION>();
                const cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();
                const cl_uint compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                const cl_uint clock_mhz = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
                const cl_ulong global_mem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                const cl_ulong max_alloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
                const cl_ulong local_mem = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
                const size_t max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
                const cl_uint max_dims = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
                const std::vector<size_t> max_wi_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
                const cl_bool image_support = device.getInfo<CL_DEVICE_IMAGE_SUPPORT>();

                std::cout << "\n  - [Device " << di << "] " << name << "\n";
                std::cout << "    Type: " << device_type_to_string(type) << "\n";
                std::cout << "    Vendor: " << vendor << "\n";
                std::cout << "    Driver: " << driver << "\n";
                std::cout << "    OpenCL: " << version << "\n";
                std::cout << "    Compute Units: " << compute_units << "\n";
                std::cout << "    Max Clock: " << clock_mhz << " MHz\n";
                std::cout << "    Global Memory: " << bytes_to_mib(global_mem) << " MiB\n";
                std::cout << "    Max Allocation: " << bytes_to_mib(max_alloc) << " MiB\n";
                std::cout << "    Local Memory: " << bytes_to_mib(local_mem) << " MiB\n";
                std::cout << "    Image Support: " << (image_support ? "Yes" : "No") << "\n";
                std::cout << "    Max Work-group Size: " << max_wg_size << "\n";
                std::cout << "    Max Work-item Dims: " << max_dims << "\n";
                std::cout << "    Max Work-item Sizes: ";
                for (size_t i = 0; i < max_wi_sizes.size(); ++i)
                {
                    if (i > 0)
                        std::cout << " x ";
                    std::cout << max_wi_sizes[i];
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

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
