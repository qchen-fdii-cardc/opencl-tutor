#include <CL/cl.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static const char* kKernelSource =
    "__kernel void vec_add(__global const float* a, __global const float* b, __global float* c) {"
    "  int gid = get_global_id(0);"
    "  c[gid] = a[gid] + b[gid];"
    "}";

static std::vector<cl_device_id> get_devices() {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
        return {};
    }
    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        return {};
    }

    for (cl_uint i = 0; i < platform_count; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count) != CL_SUCCESS ||
            device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr) == CL_SUCCESS) {
            return devices;
        }
    }
    return {};
}

static bool run_chunk(cl_device_id device,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      std::vector<float>* c) {
    const size_t n = a.size();
    const size_t bytes = n * sizeof(float);

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) return false;

    cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) return false;
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    cl_kernel kernel = clCreateKernel(program, "vec_add", &err);
    if (err != CL_SUCCESS) return false;

    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                                  const_cast<float*>(a.data()), &err);
    if (err != CL_SUCCESS || a_buf == nullptr) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                                  const_cast<float*>(b.data()), &err);
    if (err != CL_SUCCESS || b_buf == nullptr) {
        clReleaseMemObject(a_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS || c_buf == nullptr) {
        clReleaseMemObject(b_buf);
        clReleaseMemObject(a_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(c_buf);
        clReleaseMemObject(b_buf);
        clReleaseMemObject(a_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    size_t global_size = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, bytes, c->data(), 0, nullptr, nullptr);
    }

    clReleaseMemObject(c_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(a_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return err == CL_SUCCESS;
}

int main() {
    const size_t n = 1 << 18;
    std::vector<float> a(n), b(n), c(n, 0.0f), ref(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 101);
        b[i] = static_cast<float>(i % 29);
        ref[i] = a[i] + b[i];
    }

    std::vector<cl_device_id> devices = get_devices();
    if (devices.empty()) {
        std::cerr << "No usable OpenCL device found.\n";
        return EXIT_FAILURE;
    }

    const size_t used_devices = std::min<size_t>(2, devices.size());
    std::cout << "Using devices: " << used_devices << "\n";

    size_t begin = 0;
    bool ok = true;
    for (size_t d = 0; d < used_devices; ++d) {
        const size_t remain = n - begin;
        const size_t chunk = (d + 1 == used_devices) ? remain : (n / used_devices);

        std::vector<float> a_chunk(a.begin() + begin, a.begin() + begin + chunk);
        std::vector<float> b_chunk(b.begin() + begin, b.begin() + begin + chunk);
        std::vector<float> c_chunk(chunk, 0.0f);

        if (!run_chunk(devices[d], a_chunk, b_chunk, &c_chunk)) {
            ok = false;
            break;
        }

        for (size_t i = 0; i < chunk; ++i) {
            c[begin + i] = c_chunk[i];
        }

        begin += chunk;
    }

    for (size_t i = 0; ok && i < n; ++i) {
        if (std::fabs(c[i] - ref[i]) > 1e-5f) {
            ok = false;
        }
    }

    std::cout << (ok ? "Multi-device vector add verified.\n" : "Multi-device vector add failed.\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
