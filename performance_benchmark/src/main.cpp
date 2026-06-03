#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

    std::atomic<bool> g_stop_requested(false);
    std::mutex g_log_mutex;

    void signal_handler(int)
    {
        g_stop_requested.store(true);
    }

    const char *kKernelSource = R"CLC(
__kernel void heavy_math(__global float* out, uint inner_iters) {
    const size_t gid = get_global_id(0);
    float x = (float)(gid % 1024) * 0.001f + 0.5f;
    float y = (float)(gid % 2048) * 0.0005f + 0.25f;

    for (uint i = 0; i < inner_iters; ++i) {
        x = native_sin(x) * native_cos(y) + native_exp(-0.0001f * x) + native_sqrt(fabs(x) + 1.0f);
        y = native_log(fabs(y) + 1.001f) + native_tan(x * 0.001f) + 0.00001f * (float)i;
        x = mad(x, 1.000001f, y);
    }

    out[gid] = x + y;
}
)CLC";

    struct DeviceRunConfig
    {
        size_t global_work_items;
        size_t local_work_items;
        cl_uint inner_iterations;
        cl_uint launch_burst;
        int duration_seconds;
    };

    struct PlatformDevice
    {
        cl::Platform platform;
        cl::Device device;
    };

    template <typename T>
    T get_device_info_value(const cl::Device &device, cl_device_info param)
    {
        T value{};
        device.getInfo(param, &value);
        return value;
    }

    size_t clamp_global_work_items(size_t requested, cl_ulong max_alloc_bytes)
    {
        const size_t max_elements = static_cast<size_t>(max_alloc_bytes / sizeof(float));
        if (max_elements == 0)
        {
            return 1;
        }
        return requested > max_elements ? max_elements : requested;
    }

    DeviceRunConfig make_config_for_device(const cl::Device &device, int duration_seconds)
    {
        const cl_uint compute_units = get_device_info_value<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
        const cl_device_type device_type = get_device_info_value<cl_device_type>(device, CL_DEVICE_TYPE);
        const size_t max_work_group_size = get_device_info_value<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        const cl_ulong max_alloc_size = get_device_info_value<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);

        size_t local = 256;
        if (max_work_group_size < local)
        {
            local = max_work_group_size > 0 ? max_work_group_size : 1;
        }

        size_t work_items_per_cu = 32768;
        if (device_type & CL_DEVICE_TYPE_GPU)
        {
            work_items_per_cu = 131072;
        }
        else if (device_type & CL_DEVICE_TYPE_CPU)
        {
            work_items_per_cu = 8192;
        }

        size_t global = static_cast<size_t>(compute_units) * work_items_per_cu;
        if (global < local)
        {
            global = local;
        }

        global = clamp_global_work_items(global, max_alloc_size);
        if ((device_type & CL_DEVICE_TYPE_GPU) && global > (4u * 1024u * 1024u))
        {
            global = 4u * 1024u * 1024u;
        }
        global -= (global % local);
        if (global == 0)
        {
            global = local;
        }

        DeviceRunConfig cfg{};
        cfg.global_work_items = global;
        cfg.local_work_items = local;
        cfg.inner_iterations = (device_type & CL_DEVICE_TYPE_GPU) ? 8192 : 4096;
        cfg.launch_burst = (device_type & CL_DEVICE_TYPE_GPU) ? 4 : 2;
        cfg.duration_seconds = duration_seconds;
        return cfg;
    }

    std::vector<PlatformDevice> collect_targets()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        std::vector<PlatformDevice> targets;
        for (const cl::Platform &platform : platforms)
        {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (const cl::Device &device : devices)
            {
                targets.push_back({platform, device});
            }
        }

        return targets;
    }

    void run_on_device(const cl::Platform &platform, const cl::Device &device, int duration_seconds)
    {
        const std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
        const std::string device_name = device.getInfo<CL_DEVICE_NAME>();
        const cl_device_type device_type = get_device_info_value<cl_device_type>(device, CL_DEVICE_TYPE);
        const DeviceRunConfig cfg = make_config_for_device(device, duration_seconds);

        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            std::ostringstream start_msg;
            start_msg << "[START] Platform: " << platform_name
                      << " | Device: " << device_name
                      << " | Type: " << ((device_type & CL_DEVICE_TYPE_GPU) ? "GPU" : ((device_type & CL_DEVICE_TYPE_CPU) ? "CPU" : "Other"))
                      << " | Global: " << cfg.global_work_items
                      << " | Local: " << cfg.local_work_items
                      << " | InnerIters: " << cfg.inner_iterations;
            std::cout << start_msg.str() << std::endl;
        }

        cl_context_properties context_props[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(platform()),
            0};
        cl::Context context(device, context_props);
        cl::CommandQueue queue(context, device);

        cl::Program::Sources sources;
        sources.push_back({kKernelSource, std::strlen(kKernelSource)});
        cl::Program program(context, sources);
        program.build({device}, "-cl-fast-relaxed-math");

        cl::Kernel kernel(program, "heavy_math");
        cl::Buffer out_buffer(context, CL_MEM_WRITE_ONLY, cfg.global_work_items * sizeof(float));
        kernel.setArg(0, out_buffer);
        kernel.setArg(1, cfg.inner_iterations);

        const size_t global = cfg.global_work_items;
        const size_t local = cfg.local_work_items;
        const auto start = std::chrono::steady_clock::now();
        size_t launches = 0;
        bool enqueue_failed = false;

        while (!g_stop_requested.load())
        {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            if (elapsed_sec >= cfg.duration_seconds)
            {
                break;
            }

            cl_uint submitted = 0;
            for (cl_uint i = 0; i < cfg.launch_burst; ++i)
            {
                const auto submit_now = std::chrono::steady_clock::now();
                const auto submit_elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(submit_now - start).count();
                if (submit_elapsed_sec >= cfg.duration_seconds)
                {
                    break;
                }

                try
                {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
                    ++submitted;
                }
                catch (const cl::Error &e)
                {
                    std::lock_guard<std::mutex> lock(g_log_mutex);
                    std::cerr << "[ERROR] enqueueNDRangeKernel failed on " << device_name << " (" << e.err() << ")" << std::endl;
                    enqueue_failed = true;
                    break;
                }
            }

            if (enqueue_failed)
            {
                break;
            }

            if (submitted == 0)
            {
                break;
            }

            try
            {
                queue.finish();
            }
            catch (const cl::Error &e)
            {
                std::lock_guard<std::mutex> lock(g_log_mutex);
                std::cerr << "[ERROR] finish failed on " << device_name << " (" << e.err() << ")" << std::endl;
                break;
            }

            launches += submitted;
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        const double work_items_total = static_cast<double>(launches) * static_cast<double>(cfg.global_work_items) * static_cast<double>(cfg.inner_iterations);
        const double ops_est = work_items_total * 12.0;
        const double gops = (elapsed > 0.0) ? (ops_est / elapsed / 1e9) : 0.0;

        std::lock_guard<std::mutex> lock(g_log_mutex);
        std::cout << "[DONE ] Device: " << device_name
                  << " | Kernel launches: " << launches
                  << " | Elapsed(s): " << std::fixed << std::setprecision(2) << elapsed
                  << " | Estimated GOPS: " << std::fixed << std::setprecision(2) << gops
                  << std::endl;
    }

} // namespace

int main(int argc, char **argv)
{
    std::signal(SIGINT, signal_handler);

    int duration_seconds = 30;
    if (argc > 1)
    {
        try
        {
            duration_seconds = std::stoi(argv[1]);
        }
        catch (...)
        {
            std::cerr << "Invalid duration. Usage: opencl_multi_device_stress [seconds]" << std::endl;
            return 1;
        }
    }

    if (duration_seconds <= 0)
    {
        std::cerr << "Duration must be > 0" << std::endl;
        return 1;
    }

    std::vector<PlatformDevice> targets;
    try
    {
        targets = collect_targets();
    }
    catch (const cl::Error &e)
    {
        std::cerr << "OpenCL device query failed: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    }

    if (targets.empty())
    {
        std::cerr << "No OpenCL devices found" << std::endl;
        return 1;
    }

    std::cout << "Running stress kernel on " << targets.size() << " OpenCL device(s) for "
              << duration_seconds << " second(s). Press Ctrl+C to stop early." << std::endl;

    std::vector<std::thread> workers;
    workers.reserve(targets.size());
    for (const PlatformDevice &target : targets)
    {
        workers.emplace_back([target, duration_seconds]()
                             {
            try
            {
                run_on_device(target.platform, target.device, duration_seconds);
            }
            catch (const cl::Error& e)
            {
                std::lock_guard<std::mutex> lock(g_log_mutex);
                std::cerr << "[ERROR] OpenCL failure on worker: " << e.what() << " (" << e.err() << ")" << std::endl;
            }
            catch (const std::exception& e)
            {
                std::lock_guard<std::mutex> lock(g_log_mutex);
                std::cerr << "[ERROR] Worker failure: " << e.what() << std::endl;
            } });
    }

    for (std::thread &worker : workers)
    {
        worker.join();
    }

    std::cout << "All device runs completed." << std::endl;
    return 0;
}
