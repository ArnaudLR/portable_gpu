#include <webgpu/webgpu.h>
#include <gpu.h>
#include <lbm.h>
#include <inspector.h>
#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>
#include <iostream>
#include <memory>
#include <future>
#include <string>
#include <vector>

namespace gpu = ame::gpu;

int main(int, char **)
{
    try
    {
        // We can check whether there is actually an instance created
        if (!gpu::Instance::get())
        {
            std::cerr << "Could not initialize WebGPU!" << std::endl;
            return 1;
        }
        if (!glfwInit())
            throw std::runtime_error("Failed to initialize GLFW");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window(
            glfwCreateWindow(960, 540, "LBM D3Q19 - GPU volume", nullptr, nullptr), &glfwDestroyWindow);
        if (!window)
            throw std::runtime_error("Failed to create the visualization window");
        WGPUSurface surface = glfwCreateWindowWGPUSurface(gpu::Instance::get(), window.get());
        if (!surface)
            throw std::runtime_error("Failed to create the WebGPU surface");
        WGPURequestAdapterOptions adapterOptions{};
        adapterOptions.compatibleSurface = surface;
        auto adapter = gpu::Instance::Request_Adapter(adapterOptions);
        if (!adapter)
        {
            std::cerr << "Failed to obtain WebGPU adapter!" << std::endl;
            return 1;
        }
        std::cout << "WGPU adapter: " << adapter.get() << std::endl;
        // inspectAdapter(adapter.get());
        auto device = adapter.request_device({});
        if (!device)
        {
            std::cerr << "Failed to create WebGPU device!" << std::endl;
            return 1;
        }
        std::cout << "WebGPU device created successfully: " << device.get() << std::endl;
        // inspectDevice(device.get());
        run_lbm_simulation(device.get(), adapter.get(), surface, window.get());
        wgpuSurfaceRelease(surface);
        window.reset();
        glfwTerminate();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}