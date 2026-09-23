#include <webgpu/webgpu.h>
#include <gpu.h>
#include <lbm.h>
#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>
#include <iostream>
#include <memory>
#include <future>
#include <string>
#include <vector>

namespace gpu = ame::gpu;

void inspectAdapter(WGPUAdapter adapter)
{
#ifndef __EMSCRIPTEN__
    WGPULimits supportedLimits = {};
    supportedLimits.nextInChain = nullptr;

#ifdef WEBGPU_BACKEND_DAWN
    bool success = wgpuAdapterGetLimits(adapter, &supportedLimits) == WGPUStatus_Success;
#else
    bool success = wgpuAdapterGetLimits(adapter, &supportedLimits);
#endif

    if (success)
    {
        std::cout << "Adapter limits:" << std::endl;
        std::cout << " - maxTextureDimension1D: " << supportedLimits.maxTextureDimension1D << std::endl;
        std::cout << " - maxTextureDimension2D: " << supportedLimits.maxTextureDimension2D << std::endl;
        std::cout << " - maxTextureDimension3D: " << supportedLimits.maxTextureDimension3D << std::endl;
        std::cout << " - maxTextureArrayLayers: " << supportedLimits.maxTextureArrayLayers << std::endl;
    }
#endif // NOT __EMSCRIPTEN__
    std::vector<WGPUFeatureName> features;

    // Call the function a first time with a null return address, just to get
    // the entry count.
    WGPUSupportedFeatures supportedFeatures = {};
    wgpuAdapterGetFeatures(adapter, &supportedFeatures);

    std::cout << "Adapter features:" << std::endl;
    std::cout << std::hex; // Write integers as hexadecimal to ease comparison with webgpu.h literals
    for (size_t i = 0; i < supportedFeatures.featureCount; ++i)
    {
        std::cout << " - 0x" << supportedFeatures.features[i] << std::endl;
    }
    std::cout << std::dec; // Restore decimal numbers
    WGPUAdapterInfo adapterInfo = {};
    if (wgpuAdapterGetInfo(adapter, &adapterInfo) != WGPUStatus_Success)
    {
        std::cerr << "Failed to get adapter info!" << std::endl;
        return;
    }
    std::cout << "Adapter infos:" << std::endl;
    std::cout << " - vendorID: " << adapterInfo.vendorID << std::endl;
    if (adapterInfo.vendor.data)
    {
        std::cout << " - vendorName: " << std::string(adapterInfo.vendor.data, adapterInfo.vendor.length) << std::endl;
    }
    if (adapterInfo.architecture.data)
    {
        std::cout << " - architecture: " << std::string(adapterInfo.architecture.data, adapterInfo.architecture.length) << std::endl;
    }
    std::cout << " - deviceID: " << adapterInfo.deviceID << std::endl;
    if (adapterInfo.description.data)
    {
        std::cout << " - description: " << std::string(adapterInfo.description.data, adapterInfo.description.length) << std::endl;
    }
    std::cout << std::hex;
    std::cout << " - adapterType: 0x" << adapterInfo.adapterType << std::endl;
    std::cout << " - backendType: 0x" << adapterInfo.backendType << std::endl;
    std::cout << std::dec; // Restore decimal numbers
}

void inspectDevice(WGPUDevice device)
{
    WGPUSupportedFeatures supportedFeatures = {};
    wgpuDeviceGetFeatures(device, &supportedFeatures);
    std::cout << "Device features (" << supportedFeatures.featureCount << "):" << std::endl;
    std::cout << std::hex;
    for (size_t i = 0; i < supportedFeatures.featureCount; ++i)
    {
        std::cout << " - 0x" << supportedFeatures.features[i] << std::endl;
    }
    std::cout << std::dec;

    WGPULimits limits = {};
    limits.nextInChain = nullptr;

#ifdef WEBGPU_BACKEND_DAWN
    bool success = wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
#else
    bool success = wgpuDeviceGetLimits(device, &limits);
#endif

    if (success)
    {
        std::cout << "Device limits:" << std::endl;
        std::cout << " - maxTextureDimension1D: " << limits.maxTextureDimension1D << std::endl;
        std::cout << " - maxTextureDimension2D: " << limits.maxTextureDimension2D << std::endl;
        std::cout << " - maxTextureDimension3D: " << limits.maxTextureDimension3D << std::endl;
        std::cout << " - maxTextureArrayLayers: " << limits.maxTextureArrayLayers << std::endl;
        std::cout << " - maxBindGroups: " << limits.maxBindGroups << std::endl;
        std::cout << " - maxDynamicUniformBuffersPerPipelineLayout: " << limits.maxDynamicUniformBuffersPerPipelineLayout << std::endl;
        std::cout << " - maxDynamicStorageBuffersPerPipelineLayout: " << limits.maxDynamicStorageBuffersPerPipelineLayout << std::endl;
        std::cout << " - maxSampledTexturesPerShaderStage: " << limits.maxSampledTexturesPerShaderStage << std::endl;
        std::cout << " - maxSamplersPerShaderStage: " << limits.maxSamplersPerShaderStage << std::endl;
        std::cout << " - maxStorageBuffersPerShaderStage: " << limits.maxStorageBuffersPerShaderStage << std::endl;
        std::cout << " - maxStorageTexturesPerShaderStage: " << limits.maxStorageTexturesPerShaderStage << std::endl;
        std::cout << " - maxUniformBuffersPerShaderStage: " << limits.maxUniformBuffersPerShaderStage << std::endl;
        std::cout << " - maxUniformBufferBindingSize: " << limits.maxUniformBufferBindingSize << std::endl;
        std::cout << " - maxStorageBufferBindingSize: " << limits.maxStorageBufferBindingSize << std::endl;
        std::cout << " - minUniformBufferOffsetAlignment: " << limits.minUniformBufferOffsetAlignment << std::endl;
        std::cout << " - minStorageBufferOffsetAlignment: " << limits.minStorageBufferOffsetAlignment << std::endl;
        std::cout << " - maxVertexBuffers: " << limits.maxVertexBuffers << std::endl;
        std::cout << " - maxVertexAttributes: " << limits.maxVertexAttributes << std::endl;
        std::cout << " - maxVertexBufferArrayStride: " << limits.maxVertexBufferArrayStride << std::endl;
        std::cout << " - maxComputeWorkgroupStorageSize: " << limits.maxComputeWorkgroupStorageSize << std::endl;
        std::cout << " - maxComputeInvocationsPerWorkgroup: " << limits.maxComputeInvocationsPerWorkgroup << std::endl;
        std::cout << " - maxComputeWorkgroupSizeX: " << limits.maxComputeWorkgroupSizeX << std::endl;
        std::cout << " - maxComputeWorkgroupSizeY: " << limits.maxComputeWorkgroupSizeY << std::endl;
        std::cout << " - maxComputeWorkgroupSizeZ: " << limits.maxComputeWorkgroupSizeZ << std::endl;
        std::cout << " - maxComputeWorkgroupsPerDimension: " << limits.maxComputeWorkgroupsPerDimension << std::endl;
    }
}
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
        WGPUSurface surface = glfwGetWGPUSurface(gpu::Instance::get(), window.get());
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