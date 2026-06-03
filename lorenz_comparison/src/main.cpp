#ifndef NOMINMAX
#define NOMINMAX
#endif
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    const char *kLorenzKernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline void lorenz_rhs(const double x, const double y, const double z,
                       const double sigma, const double rho, const double beta,
                       double* dx, double* dy, double* dz) {
    *dx = sigma * (y - x);
    *dy = x * (rho - z) - y;
    *dz = x * y - beta * z;
}

inline void rk4_step(double* x, double* y, double* z,
                     const double dt,
                     const double sigma, const double rho, const double beta) {
    double k1x, k1y, k1z;
    lorenz_rhs(*x, *y, *z, sigma, rho, beta, &k1x, &k1y, &k1z);

    const double x2 = *x + 0.5 * dt * k1x;
    const double y2 = *y + 0.5 * dt * k1y;
    const double z2 = *z + 0.5 * dt * k1z;
    double k2x, k2y, k2z;
    lorenz_rhs(x2, y2, z2, sigma, rho, beta, &k2x, &k2y, &k2z);

    const double x3 = *x + 0.5 * dt * k2x;
    const double y3 = *y + 0.5 * dt * k2y;
    const double z3 = *z + 0.5 * dt * k2z;
    double k3x, k3y, k3z;
    lorenz_rhs(x3, y3, z3, sigma, rho, beta, &k3x, &k3y, &k3z);

    const double x4 = *x + dt * k3x;
    const double y4 = *y + dt * k3y;
    const double z4 = *z + dt * k3z;
    double k4x, k4y, k4z;
    lorenz_rhs(x4, y4, z4, sigma, rho, beta, &k4x, &k4y, &k4z);

    *x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    *y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    *z += (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
}

__kernel void lorenz_benchmark(
    __global const double* in_x,
    __global const double* in_y,
    __global const double* in_z,
    __global double* out_x,
    __global double* out_y,
    __global double* out_z,
    const uint steps,
    const double dt,
    const double sigma,
    const double rho,
    const double beta) {
    const uint gid = get_global_id(0);
    double x = in_x[gid];
    double y = in_y[gid];
    double z = in_z[gid];

    for (uint i = 0; i < steps; ++i) {
        rk4_step(&x, &y, &z, dt, sigma, rho, beta);
    }

    out_x[gid] = x;
    out_y[gid] = y;
    out_z[gid] = z;
}

__kernel void lorenz_history(
    __global double* out_x_hist,
    __global double* out_y_hist,
    __global double* out_z_hist,
    const uint steps,
    const double dt,
    const double sigma,
    const double rho,
    const double beta,
    const double x0,
    const double y0,
    const double z0) {
    double x = x0;
    double y = y0;
    double z = z0;

    out_x_hist[0] = x;
    out_y_hist[0] = y;
    out_z_hist[0] = z;

    for (uint i = 0; i < steps; ++i) {
        rk4_step(&x, &y, &z, dt, sigma, rho, beta);
        out_x_hist[i + 1] = x;
        out_y_hist[i + 1] = y;
        out_z_hist[i + 1] = z;
    }
}
)CLC";

    struct LorenzParams
    {
        double sigma = 10.0;
        double rho = 28.0;
        double beta = 8.0 / 3.0;
        double dt = 1e-3;
        int steps = 20000;
        int trajectories = 8192;
    };

    struct BenchmarkResult
    {
        double seconds = 0.0;
        std::vector<double> out_x;
        std::vector<double> out_y;
        std::vector<double> out_z;
    };

    struct HistoryResult
    {
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
    };

    void lorenz_rhs_cpu(const double x, const double y, const double z,
                        const LorenzParams &p,
                        double *dx, double *dy, double *dz)
    {
        *dx = p.sigma * (y - x);
        *dy = x * (p.rho - z) - y;
        *dz = x * y - p.beta * z;
    }

    void rk4_step_cpu(double *x, double *y, double *z, const LorenzParams &p)
    {
        double k1x, k1y, k1z;
        lorenz_rhs_cpu(*x, *y, *z, p, &k1x, &k1y, &k1z);

        const double x2 = *x + 0.5 * p.dt * k1x;
        const double y2 = *y + 0.5 * p.dt * k1y;
        const double z2 = *z + 0.5 * p.dt * k1z;
        double k2x, k2y, k2z;
        lorenz_rhs_cpu(x2, y2, z2, p, &k2x, &k2y, &k2z);

        const double x3 = *x + 0.5 * p.dt * k2x;
        const double y3 = *y + 0.5 * p.dt * k2y;
        const double z3 = *z + 0.5 * p.dt * k2z;
        double k3x, k3y, k3z;
        lorenz_rhs_cpu(x3, y3, z3, p, &k3x, &k3y, &k3z);

        const double x4 = *x + p.dt * k3x;
        const double y4 = *y + p.dt * k3y;
        const double z4 = *z + p.dt * k3z;
        double k4x, k4y, k4z;
        lorenz_rhs_cpu(x4, y4, z4, p, &k4x, &k4y, &k4z);

        *x += (p.dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        *y += (p.dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        *z += (p.dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }

    std::vector<cl::Device> get_all_gpu_devices()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        std::vector<cl::Device> all_gpus;
        for (const cl::Platform &platform : platforms)
        {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            all_gpus.insert(all_gpus.end(), devices.begin(), devices.end());
        }
        return all_gpus;
    }

    cl::Device select_gpu_device_prefer_nvidia()
    {
        std::vector<cl::Device> gpus = get_all_gpu_devices();
        if (gpus.empty())
        {
            throw std::runtime_error("No OpenCL GPU device found.");
        }

        for (const cl::Device &dev : gpus)
        {
            const std::string vendor = dev.getInfo<CL_DEVICE_VENDOR>();
            const std::string name = dev.getInfo<CL_DEVICE_NAME>();
            if (vendor.find("NVIDIA") != std::string::npos || name.find("NVIDIA") != std::string::npos)
            {
                return dev;
            }
        }

        std::cerr << "[WARN] NVIDIA GPU not found. Falling back to first GPU: "
                  << gpus[0].getInfo<CL_DEVICE_NAME>() << "\n";
        return gpus[0];
    }

    BenchmarkResult run_cpu_benchmark(const LorenzParams &p,
                                      const std::vector<double> &init_x,
                                      const std::vector<double> &init_y,
                                      const std::vector<double> &init_z)
    {
        BenchmarkResult result;
        result.out_x = init_x;
        result.out_y = init_y;
        result.out_z = init_z;

        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int idx = 0; idx < p.trajectories; ++idx)
        {
            double x = result.out_x[static_cast<size_t>(idx)];
            double y = result.out_y[static_cast<size_t>(idx)];
            double z = result.out_z[static_cast<size_t>(idx)];

            for (int s = 0; s < p.steps; ++s)
            {
                rk4_step_cpu(&x, &y, &z, p);
            }

            result.out_x[static_cast<size_t>(idx)] = x;
            result.out_y[static_cast<size_t>(idx)] = y;
            result.out_z[static_cast<size_t>(idx)] = z;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        result.seconds = std::chrono::duration<double>(t1 - t0).count();
        return result;
    }

    BenchmarkResult run_opencl_benchmark(const LorenzParams &p,
                                         const std::vector<double> &init_x,
                                         const std::vector<double> &init_y,
                                         const std::vector<double> &init_z,
                                         const cl::Context &context,
                                         const cl::Device &device,
                                         const cl::Program &program)
    {
        BenchmarkResult result;
        result.out_x.resize(init_x.size());
        result.out_y.resize(init_y.size());
        result.out_z.resize(init_z.size());

        cl::CommandQueue queue(context, device);

        cl::Buffer in_x_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(double) * init_x.size(),
                            const_cast<double *>(init_x.data()));
        cl::Buffer in_y_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(double) * init_y.size(),
                            const_cast<double *>(init_y.data()));
        cl::Buffer in_z_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(double) * init_z.size(),
                            const_cast<double *>(init_z.data()));

        cl::Buffer out_x_buf(context, CL_MEM_WRITE_ONLY, sizeof(double) * result.out_x.size());
        cl::Buffer out_y_buf(context, CL_MEM_WRITE_ONLY, sizeof(double) * result.out_y.size());
        cl::Buffer out_z_buf(context, CL_MEM_WRITE_ONLY, sizeof(double) * result.out_z.size());

        cl::Kernel kernel(program, "lorenz_benchmark");
        kernel.setArg(0, in_x_buf);
        kernel.setArg(1, in_y_buf);
        kernel.setArg(2, in_z_buf);
        kernel.setArg(3, out_x_buf);
        kernel.setArg(4, out_y_buf);
        kernel.setArg(5, out_z_buf);
        kernel.setArg(6, static_cast<cl_uint>(p.steps));
        kernel.setArg(7, p.dt);
        kernel.setArg(8, p.sigma);
        kernel.setArg(9, p.rho);
        kernel.setArg(10, p.beta);

        const auto t0 = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(static_cast<size_t>(p.trajectories)), cl::NullRange);
        queue.finish();
        const auto t1 = std::chrono::high_resolution_clock::now();
        result.seconds = std::chrono::duration<double>(t1 - t0).count();

        queue.enqueueReadBuffer(out_x_buf, CL_TRUE, 0, sizeof(double) * result.out_x.size(), result.out_x.data());
        queue.enqueueReadBuffer(out_y_buf, CL_TRUE, 0, sizeof(double) * result.out_y.size(), result.out_y.data());
        queue.enqueueReadBuffer(out_z_buf, CL_TRUE, 0, sizeof(double) * result.out_z.size(), result.out_z.data());

        return result;
    }

    HistoryResult run_cpu_history(const LorenzParams &p, double x0, double y0, double z0)
    {
        HistoryResult h;
        h.x.resize(static_cast<size_t>(p.steps) + 1);
        h.y.resize(static_cast<size_t>(p.steps) + 1);
        h.z.resize(static_cast<size_t>(p.steps) + 1);

        double x = x0;
        double y = y0;
        double z = z0;
        h.x[0] = x;
        h.y[0] = y;
        h.z[0] = z;

        for (int i = 0; i < p.steps; ++i)
        {
            rk4_step_cpu(&x, &y, &z, p);
            const size_t idx = static_cast<size_t>(i) + 1;
            h.x[idx] = x;
            h.y[idx] = y;
            h.z[idx] = z;
        }

        return h;
    }

    HistoryResult run_opencl_history(const LorenzParams &p,
                                     double x0,
                                     double y0,
                                     double z0,
                                     const cl::Context &context,
                                     const cl::Device &device,
                                     const cl::Program &program)
    {
        HistoryResult h;
        h.x.resize(static_cast<size_t>(p.steps) + 1);
        h.y.resize(static_cast<size_t>(p.steps) + 1);
        h.z.resize(static_cast<size_t>(p.steps) + 1);

        cl::CommandQueue queue(context, device);

        cl::Buffer x_hist_buf(context, CL_MEM_WRITE_ONLY, sizeof(double) * h.x.size());
        cl::Buffer y_hist_buf(context, CL_MEM_WRITE_ONLY, sizeof(double) * h.y.size());
        cl::Buffer z_hist_buf(context, CL_MEM_WRITE_ONLY, sizeof(double) * h.z.size());

        cl::Kernel kernel(program, "lorenz_history");
        kernel.setArg(0, x_hist_buf);
        kernel.setArg(1, y_hist_buf);
        kernel.setArg(2, z_hist_buf);
        kernel.setArg(3, static_cast<cl_uint>(p.steps));
        kernel.setArg(4, p.dt);
        kernel.setArg(5, p.sigma);
        kernel.setArg(6, p.rho);
        kernel.setArg(7, p.beta);
        kernel.setArg(8, x0);
        kernel.setArg(9, y0);
        kernel.setArg(10, z0);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
        queue.finish();

        queue.enqueueReadBuffer(x_hist_buf, CL_TRUE, 0, sizeof(double) * h.x.size(), h.x.data());
        queue.enqueueReadBuffer(y_hist_buf, CL_TRUE, 0, sizeof(double) * h.y.size(), h.y.data());
        queue.enqueueReadBuffer(z_hist_buf, CL_TRUE, 0, sizeof(double) * h.z.size(), h.z.data());

        return h;
    }

    cl::Program build_program_for_device(const cl::Context &context, const cl::Device &device)
    {
        cl::Program::Sources sources;
        sources.push_back({kLorenzKernelSource, std::strlen(kLorenzKernelSource)});
        cl::Program program(context, sources);
        program.build({device}, "");
        return program;
    }

    void write_validation_csv(const std::string &csv_path,
                              const LorenzParams &p,
                              const HistoryResult &cpu,
                              const HistoryResult &gpu)
    {
        std::ofstream out(csv_path.c_str());
        if (!out)
        {
            throw std::runtime_error("Failed to open output CSV: " + csv_path);
        }

        out << "step,time,cpu_x,cpu_y,cpu_z,gpu_x,gpu_y,gpu_z,abs_dx,abs_dy,abs_dz,max_abs_diff\n";
        out << std::setprecision(17);

        for (int i = 0; i <= p.steps; ++i)
        {
            const size_t idx = static_cast<size_t>(i);
            const double dx = std::fabs(cpu.x[idx] - gpu.x[idx]);
            const double dy = std::fabs(cpu.y[idx] - gpu.y[idx]);
            const double dz = std::fabs(cpu.z[idx] - gpu.z[idx]);
            const double maxd = std::max(dx, std::max(dy, dz));
            const double time = p.dt * static_cast<double>(i);
            out << i << "," << time << ","
                << cpu.x[idx] << "," << cpu.y[idx] << "," << cpu.z[idx] << ","
                << gpu.x[idx] << "," << gpu.y[idx] << "," << gpu.z[idx] << ","
                << dx << "," << dy << "," << dz << "," << maxd << "\n";
        }
    }

    double compute_max_final_diff(const BenchmarkResult &cpu, const BenchmarkResult &gpu)
    {
        double maxd = 0.0;
        for (size_t i = 0; i < cpu.out_x.size(); ++i)
        {
            const double dx = std::fabs(cpu.out_x[i] - gpu.out_x[i]);
            const double dy = std::fabs(cpu.out_y[i] - gpu.out_y[i]);
            const double dz = std::fabs(cpu.out_z[i] - gpu.out_z[i]);
            maxd = std::max(maxd, std::max(dx, std::max(dy, dz)));
        }
        return maxd;
    }

    double compute_history_max_diff(const HistoryResult &cpu, const HistoryResult &gpu)
    {
        double maxd = 0.0;
        for (size_t i = 0; i < cpu.x.size(); ++i)
        {
            const double dx = std::fabs(cpu.x[i] - gpu.x[i]);
            const double dy = std::fabs(cpu.y[i] - gpu.y[i]);
            const double dz = std::fabs(cpu.z[i] - gpu.z[i]);
            maxd = std::max(maxd, std::max(dx, std::max(dy, dz)));
        }
        return maxd;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        LorenzParams p;
        if (argc > 1)
        {
            p.steps = std::max(1, std::stoi(argv[1]));
        }
        if (argc > 2)
        {
            p.trajectories = std::max(1, std::stoi(argv[2]));
        }

        const std::string csv_path = "lorenz_comparison/lorenz_validation.csv";

        std::vector<double> init_x(static_cast<size_t>(p.trajectories));
        std::vector<double> init_y(static_cast<size_t>(p.trajectories));
        std::vector<double> init_z(static_cast<size_t>(p.trajectories));
        for (int i = 0; i < p.trajectories; ++i)
        {
            const double f = static_cast<double>(i);
            init_x[static_cast<size_t>(i)] = -8.0 + 0.0001 * f;
            init_y[static_cast<size_t>(i)] = 8.0 - 0.0001 * f;
            init_z[static_cast<size_t>(i)] = 27.0 + 0.00005 * f;
        }

        cl::Device device = select_gpu_device_prefer_nvidia();
        std::cout << "OpenCL device selected: " << device.getInfo<CL_DEVICE_NAME>()
                  << " | Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";

        cl::Context context(device);
        cl::Program program = build_program_for_device(context, device);

        const BenchmarkResult cpu = run_cpu_benchmark(p, init_x, init_y, init_z);
        const BenchmarkResult gpu = run_opencl_benchmark(p, init_x, init_y, init_z, context, device, program);
        const double final_max_diff = compute_max_final_diff(cpu, gpu);

        const double x0 = -8.0;
        const double y0 = 8.0;
        const double z0 = 27.0;
        const HistoryResult cpu_hist = run_cpu_history(p, x0, y0, z0);
        const HistoryResult gpu_hist = run_opencl_history(p, x0, y0, z0, context, device, program);
        const double history_max_diff = compute_history_max_diff(cpu_hist, gpu_hist);

        write_validation_csv(csv_path, p, cpu_hist, gpu_hist);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "CPU benchmark time (s): " << cpu.seconds << "\n";
        std::cout << "OpenCL benchmark time (s): " << gpu.seconds << "\n";
        std::cout << "Speedup (CPU/OpenCL): " << (cpu.seconds / std::max(gpu.seconds, 1e-12)) << "x\n";
        std::cout << std::scientific;
        std::cout << "Max abs diff (benchmark final states): " << final_max_diff << "\n";
        std::cout << "Max abs diff (history CSV): " << history_max_diff << "\n";
        std::cout << "Validation CSV written: " << csv_path << "\n";
        std::cout << "Args: [steps] [trajectories], current=" << p.steps << ", " << p.trajectories << "\n";
        return 0;
    }
    catch (const cl::Error &e)
    {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
