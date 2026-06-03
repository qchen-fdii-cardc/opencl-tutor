#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

static const char* kKernelSource =
    "__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;"
    "__kernel void invert_image(read_only image2d_t src, write_only image2d_t dst) {"
    "  int2 coord = (int2)(get_global_id(0), get_global_id(1));"
    "  uint4 px = read_imageui(src, smp, coord);"
    "  px.x = 255 - px.x;"
    "  px.y = 255 - px.y;"
    "  px.z = 255 - px.z;"
    "  write_imageui(dst, coord, px);"
    "}";

static bool pick_first_device(cl_device_id* out_device) {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) return false;
    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) return false;
    for (cl_uint i = 0; i < platform_count; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count) != CL_SUCCESS ||
            device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr) == CL_SUCCESS) {
            *out_device = devices[0];
            return true;
        }
    }
    return false;
}

int main() {
    const size_t width = 16;
    const size_t height = 16;
    const size_t pixel_bytes = 4;
    const size_t bytes = width * height * pixel_bytes;

    std::vector<unsigned char> src(bytes, 0);
    std::vector<unsigned char> dst(bytes, 0);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const size_t idx = (y * width + x) * pixel_bytes;
            src[idx + 0] = static_cast<unsigned char>((x * 13) & 0xFF);
            src[idx + 1] = static_cast<unsigned char>((y * 17) & 0xFF);
            src[idx + 2] = static_cast<unsigned char>(((x + y) * 7) & 0xFF);
            src[idx + 3] = 255;
        }
    }

    cl_device_id device = nullptr;
    if (!pick_first_device(&device)) {
        std::cerr << "No usable OpenCL device found.\n";
        return EXIT_FAILURE;
    }

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_kernel kernel = clCreateKernel(program, "invert_image", &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNSIGNED_INT8;

    cl_image_desc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;

    cl_mem src_image = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     &format, &desc, src.data(), &err);
    if (err != CL_SUCCESS || src_image == nullptr) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
    cl_mem dst_image = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, nullptr, &err);
    if (err != CL_SUCCESS || dst_image == nullptr) {
        clReleaseMemObject(src_image);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_image);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(dst_image);
        clReleaseMemObject(src_image);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    size_t global_size[2] = {width, height};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    if (err == CL_SUCCESS) {
        const size_t origin[3] = {0, 0, 0};
        const size_t region[3] = {width, height, 1};
        err = clEnqueueReadImage(queue, dst_image, CL_TRUE, origin, region, 0, 0, dst.data(), 0, nullptr, nullptr);
    }

    bool ok = (err == CL_SUCCESS);
    for (size_t i = 0; ok && i < bytes; i += pixel_bytes) {
        if (dst[i + 0] != static_cast<unsigned char>(255 - src[i + 0]) ||
            dst[i + 1] != static_cast<unsigned char>(255 - src[i + 1]) ||
            dst[i + 2] != static_cast<unsigned char>(255 - src[i + 2]) ||
            dst[i + 3] != src[i + 3]) {
            ok = false;
        }
    }

    std::cout << (ok ? "Image sampler task verified.\n" : "Image sampler task failed.\n");

    clReleaseMemObject(dst_image);
    clReleaseMemObject(src_image);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
