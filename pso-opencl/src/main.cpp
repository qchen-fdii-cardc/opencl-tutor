#ifndef NOMINMAX
#define NOMINMAX
#endif
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    const char *kPsoKernels = R"CLC(
inline uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline float rand01(uint base) {
    const uint h = hash_u32(base);
    return ((float)(h & 0x00FFFFFFu)) / 16777216.0f;
}

inline float rastrigin_value(const __global float* x, const int offset, const int dims) {
    const float pi = 3.14159265358979323846f;
    float sum = 10.0f * (float)dims;
    for (int d = 0; d < dims; ++d) {
        const float v = x[offset + d];
        sum += v * v - 10.0f * cos(2.0f * pi * v);
    }
    return sum;
}

__kernel void evaluate_and_update_pbest(
    __global const float* positions,
    __global float* pbest_positions,
    __global float* pbest_fitness,
    const int dims,
    const int particle_count) {
    const int i = (int)get_global_id(0);
    if (i >= particle_count) return;

    const int offset = i * dims;
    const float fit = rastrigin_value(positions, offset, dims);

    if (fit < pbest_fitness[i]) {
        pbest_fitness[i] = fit;
        for (int d = 0; d < dims; ++d) {
            pbest_positions[offset + d] = positions[offset + d];
        }
    }
}

__kernel void update_particles(
    __global float* positions,
    __global float* velocities,
    __global const float* pbest_positions,
    __global const float* gbest_position,
    const int dims,
    const int particle_count,
    const float inertia,
    const float c1,
    const float c2,
    const float x_min,
    const float x_max,
    const int iter_seed) {
    const int i = (int)get_global_id(0);
    if (i >= particle_count) return;

    const int offset = i * dims;
    for (int d = 0; d < dims; ++d) {
        const uint seed = (uint)(iter_seed * 73856093 + i * 19349663 + d * 83492791);
        const float r1 = rand01(seed);
        const float r2 = rand01(seed ^ 0xA5A5A5A5u);

        const float x = positions[offset + d];
        const float v = velocities[offset + d];
        const float p = pbest_positions[offset + d];
        const float g = gbest_position[d];

        float v_new = inertia * v + c1 * r1 * (p - x) + c2 * r2 * (g - x);
        float x_new = x + v_new;

        if (x_new < x_min) {
            x_new = x_min;
            v_new = 0.0f;
        } else if (x_new > x_max) {
            x_new = x_max;
            v_new = 0.0f;
        }

        velocities[offset + d] = v_new;
        positions[offset + d] = x_new;
    }
}
)CLC";

    struct PsoConfig
    {
        int particle_count = 8192;
        int dimensions = 32;
        int iterations = 400;
        float inertia = 0.72f;
        float c1 = 1.49f;
        float c2 = 1.49f;
        float min_bound = -5.12f;
        float max_bound = 5.12f;
        unsigned int seed = 1337u;
    };

    struct PsoResult
    {
        float best_fitness = std::numeric_limits<float>::infinity();
        std::vector<float> best_position;
        std::vector<float> history;
        double elapsed_seconds = 0.0;
    };

    class OpenClPsoFramework
    {
    public:
        explicit OpenClPsoFramework(const PsoConfig &cfg)
            : cfg_(cfg),
              total_values_(static_cast<size_t>(cfg_.particle_count) * static_cast<size_t>(cfg_.dimensions)) {}

        PsoResult run()
        {
            pick_device();
            build_program();
            allocate_and_initialize();
            return optimize();
        }

        const std::string &device_name() const { return device_name_; }
        const std::string &vendor_name() const { return vendor_name_; }

    private:
        void pick_device()
        {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);

            std::vector<cl::Device> gpus;
            for (const cl::Platform &platform : platforms)
            {
                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                gpus.insert(gpus.end(), devices.begin(), devices.end());
            }

            if (gpus.empty())
            {
                throw std::runtime_error("No OpenCL GPU found for PSO.");
            }

            size_t chosen = 0;
            for (size_t i = 0; i < gpus.size(); ++i)
            {
                const std::string vendor = gpus[i].getInfo<CL_DEVICE_VENDOR>();
                const std::string name = gpus[i].getInfo<CL_DEVICE_NAME>();
                if (vendor.find("NVIDIA") != std::string::npos || name.find("NVIDIA") != std::string::npos)
                {
                    chosen = i;
                    break;
                }
            }

            device_ = gpus[chosen];
            device_name_ = device_.getInfo<CL_DEVICE_NAME>();
            vendor_name_ = device_.getInfo<CL_DEVICE_VENDOR>();
            context_ = cl::Context(device_);
            queue_ = cl::CommandQueue(context_, device_);
        }

        void build_program()
        {
            cl::Program::Sources sources;
            sources.push_back({kPsoKernels, std::strlen(kPsoKernels)});
            program_ = cl::Program(context_, sources);
            program_.build({device_}, "-cl-mad-enable");

            eval_kernel_ = cl::Kernel(program_, "evaluate_and_update_pbest");
            update_kernel_ = cl::Kernel(program_, "update_particles");
        }

        void allocate_and_initialize()
        {
            std::mt19937 rng(cfg_.seed);
            std::uniform_real_distribution<float> pos_dist(cfg_.min_bound, cfg_.max_bound);
            std::uniform_real_distribution<float> vel_dist(-0.2f, 0.2f);

            h_positions_.resize(total_values_);
            h_velocities_.resize(total_values_);
            h_pbest_positions_.resize(total_values_);
            h_pbest_fitness_.assign(static_cast<size_t>(cfg_.particle_count), std::numeric_limits<float>::infinity());
            h_gbest_position_.assign(static_cast<size_t>(cfg_.dimensions), 0.0f);

            for (size_t i = 0; i < total_values_; ++i)
            {
                h_positions_[i] = pos_dist(rng);
                h_velocities_[i] = vel_dist(rng);
                h_pbest_positions_[i] = h_positions_[i];
            }

            d_positions_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * h_positions_.size(), h_positions_.data());
            d_velocities_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * h_velocities_.size(), h_velocities_.data());
            d_pbest_positions_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            sizeof(float) * h_pbest_positions_.size(), h_pbest_positions_.data());
            d_pbest_fitness_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * h_pbest_fitness_.size(), h_pbest_fitness_.data());
            d_gbest_position_ = cl::Buffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * h_gbest_position_.size(), h_gbest_position_.data());
        }

        PsoResult optimize()
        {
            PsoResult result;
            result.history.reserve(static_cast<size_t>(cfg_.iterations));
            result.best_position.assign(static_cast<size_t>(cfg_.dimensions), 0.0f);

            eval_kernel_.setArg(0, d_positions_);
            eval_kernel_.setArg(1, d_pbest_positions_);
            eval_kernel_.setArg(2, d_pbest_fitness_);
            eval_kernel_.setArg(3, cfg_.dimensions);
            eval_kernel_.setArg(4, cfg_.particle_count);

            update_kernel_.setArg(0, d_positions_);
            update_kernel_.setArg(1, d_velocities_);
            update_kernel_.setArg(2, d_pbest_positions_);
            update_kernel_.setArg(3, d_gbest_position_);
            update_kernel_.setArg(4, cfg_.dimensions);
            update_kernel_.setArg(5, cfg_.particle_count);
            update_kernel_.setArg(6, cfg_.inertia);
            update_kernel_.setArg(7, cfg_.c1);
            update_kernel_.setArg(8, cfg_.c2);
            update_kernel_.setArg(9, cfg_.min_bound);
            update_kernel_.setArg(10, cfg_.max_bound);

            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < cfg_.iterations; ++iter)
            {
                queue_.enqueueNDRangeKernel(eval_kernel_, cl::NullRange,
                                            cl::NDRange(static_cast<size_t>(cfg_.particle_count)), cl::NullRange);
                queue_.finish();

                queue_.enqueueReadBuffer(d_pbest_fitness_, CL_TRUE, 0,
                                         sizeof(float) * h_pbest_fitness_.size(), h_pbest_fitness_.data());

                int best_idx = 0;
                float best_fit = h_pbest_fitness_[0];
                for (int i = 1; i < cfg_.particle_count; ++i)
                {
                    if (h_pbest_fitness_[static_cast<size_t>(i)] < best_fit)
                    {
                        best_fit = h_pbest_fitness_[static_cast<size_t>(i)];
                        best_idx = i;
                    }
                }

                if (best_fit < result.best_fitness)
                {
                    result.best_fitness = best_fit;
                    const size_t pos_offset = static_cast<size_t>(best_idx) * static_cast<size_t>(cfg_.dimensions);
                    queue_.enqueueReadBuffer(d_pbest_positions_, CL_TRUE,
                                             sizeof(float) * pos_offset,
                                             sizeof(float) * result.best_position.size(),
                                             result.best_position.data());
                    h_gbest_position_ = result.best_position;
                    queue_.enqueueWriteBuffer(d_gbest_position_, CL_TRUE, 0,
                                              sizeof(float) * h_gbest_position_.size(), h_gbest_position_.data());
                }

                result.history.push_back(result.best_fitness);

                update_kernel_.setArg(11, iter + 1);
                queue_.enqueueNDRangeKernel(update_kernel_, cl::NullRange,
                                            cl::NDRange(static_cast<size_t>(cfg_.particle_count)), cl::NullRange);
                queue_.finish();
            }

            const auto t1 = std::chrono::high_resolution_clock::now();
            result.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
            return result;
        }

    private:
        PsoConfig cfg_;
        size_t total_values_;

        cl::Device device_;
        std::string device_name_;
        std::string vendor_name_;
        cl::Context context_;
        cl::CommandQueue queue_;
        cl::Program program_;
        cl::Kernel eval_kernel_;
        cl::Kernel update_kernel_;

        std::vector<float> h_positions_;
        std::vector<float> h_velocities_;
        std::vector<float> h_pbest_positions_;
        std::vector<float> h_pbest_fitness_;
        std::vector<float> h_gbest_position_;

        cl::Buffer d_positions_;
        cl::Buffer d_velocities_;
        cl::Buffer d_pbest_positions_;
        cl::Buffer d_pbest_fitness_;
        cl::Buffer d_gbest_position_;
    };

    void write_history_csv(const std::string &path, const std::vector<float> &history)
    {
        std::ofstream out(path.c_str());
        if (!out)
        {
            throw std::runtime_error("Unable to open CSV output: " + path);
        }

        out << "iteration,best_fitness\n";
        out << std::setprecision(9);
        for (size_t i = 0; i < history.size(); ++i)
        {
            out << i << "," << history[i] << "\n";
        }
    }

    PsoConfig parse_args(int argc, char **argv)
    {
        PsoConfig cfg;
        if (argc > 1)
            cfg.particle_count = std::max(1, std::stoi(argv[1]));
        if (argc > 2)
            cfg.dimensions = std::max(1, std::stoi(argv[2]));
        if (argc > 3)
            cfg.iterations = std::max(1, std::stoi(argv[3]));
        return cfg;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const PsoConfig cfg = parse_args(argc, argv);
        OpenClPsoFramework framework(cfg);
        const PsoResult result = framework.run();

        const std::string csv_path = "pso_convergence.csv";
        write_history_csv(csv_path, result.history);

        const double evals = static_cast<double>(cfg.particle_count) * static_cast<double>(cfg.iterations);
        const double evals_per_sec = evals / std::max(result.elapsed_seconds, 1e-12);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Device: " << framework.device_name() << " | Vendor: " << framework.vendor_name() << "\n";
        std::cout << "Particles: " << cfg.particle_count << " | Dimensions: " << cfg.dimensions
                  << " | Iterations: " << cfg.iterations << "\n";
        std::cout << "Best fitness (Rastrigin): " << result.best_fitness << "\n";
        std::cout << "Elapsed (s): " << result.elapsed_seconds << "\n";
        std::cout << "Particle evaluations/s: " << evals_per_sec << "\n";
        std::cout << "Convergence CSV: " << csv_path << "\n";

        std::cout << "Best position (first 8 dims): ";
        const size_t show = std::min<size_t>(8, result.best_position.size());
        for (size_t i = 0; i < show; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << result.best_position[i];
        }
        std::cout << "\n";

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
