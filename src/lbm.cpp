#include "lbm.h"
#include <gpu.h>
#include <GLFW/glfw3.h>
#if defined(WEBGPU_BACKEND_WGPU)
// wgpu-native extensions (wgpuDevicePoll, ...). This header is only shipped by
// the wgpu-native distribution, so it must not be included with Dawn/Emscripten.
#include <webgpu/wgpu.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef LBM_SHADER_PATH
#define LBM_SHADER_PATH "src/lbm_kernels.wgsl"
#endif
#ifndef LBM_VISUAL_SHADER_PATH
#define LBM_VISUAL_SHADER_PATH "src/visualization.wgsl"
#endif

namespace
{
    // A compact, x-major D3Q19 population grid. Periodic boundaries are used in
    // x/z; the kernel applies bounce-back at y walls and around a sphere.
    constexpr std::uint32_t kWidth = 64;
    constexpr std::uint32_t kHeight = 32;
    constexpr std::uint32_t kDepth = 32;
    constexpr std::uint32_t kDirections = 19;
    constexpr std::uint32_t kSteps = 200;
    constexpr float kInitialVelocity = 0.04F;
    constexpr float kRelaxation = 1.0F;
    constexpr std::uint32_t kWorkgroupSize = 4;

    constexpr std::array<std::array<int, 3>, kDirections> kLatticeDirections{{
        {{0, 0, 0}},
        {{1, 0, 0}}, {{-1, 0, 0}},
        {{0, 1, 0}}, {{0, -1, 0}},
        {{0, 0, 1}}, {{0, 0, -1}},
        {{1, 1, 0}}, {{-1, -1, 0}},
        {{1, -1, 0}}, {{-1, 1, 0}},
        {{1, 0, 1}}, {{-1, 0, -1}},
        {{1, 0, -1}}, {{-1, 0, 1}},
        {{0, 1, 1}}, {{0, -1, -1}},
        {{0, 1, -1}}, {{0, -1, 1}}
    }};
    constexpr std::array<float, kDirections> kWeights{{
        1.0F / 3.0F,
        1.0F / 18.0F, 1.0F / 18.0F, 1.0F / 18.0F, 1.0F / 18.0F, 1.0F / 18.0F, 1.0F / 18.0F,
        1.0F / 36.0F, 1.0F / 36.0F, 1.0F / 36.0F, 1.0F / 36.0F,
        1.0F / 36.0F, 1.0F / 36.0F, 1.0F / 36.0F, 1.0F / 36.0F,
        1.0F / 36.0F, 1.0F / 36.0F, 1.0F / 36.0F, 1.0F / 36.0F
    }};

    struct Parameters
    {
        std::array<std::uint32_t, 4> dimensions;
        float omega;
        float initialVelocity;
        float obstacleRadius;
        float padding;
    };
    static_assert(sizeof(Parameters) == 32);

    template <typename Handle, void (*Release)(Handle)>
    struct WgpuDeleter
    {
        void operator()(Handle handle) const noexcept
        {
            if (handle)
                Release(handle);
        }
    };

    template <typename Handle, void (*Release)(Handle)>
    using WgpuHandle = std::unique_ptr<std::remove_pointer_t<Handle>, WgpuDeleter<Handle, Release>>;

    float equilibrium(std::size_t direction, float density, float ux)
    {
        const auto& c = kLatticeDirections[direction];
        const float cu = static_cast<float>(c[0]) * ux;
        return kWeights[direction] * density * (1.0F + 3.0F * cu + 4.5F * cu * cu - 1.5F * ux * ux);
    }

    std::vector<float> make_initial_populations()
    {
        const auto siteCount = static_cast<std::size_t>(kWidth) * kHeight * kDepth;
        std::vector<float> populations(siteCount * kDirections);
        const float obstacleX = static_cast<float>(kWidth) * 0.25F;
        const float obstacleY = static_cast<float>(kHeight) * 0.5F;
        const float obstacleZ = static_cast<float>(kDepth) * 0.5F;
        const float obstacleRadius = static_cast<float>(kHeight) * 0.15F;
        const float radiusSquared = obstacleRadius * obstacleRadius;

        for (std::uint32_t z = 0; z < kDepth; ++z)
        {
            for (std::uint32_t y = 0; y < kHeight; ++y)
            {
                for (std::uint32_t x = 0; x < kWidth; ++x)
                {
                    const float dx = static_cast<float>(x) - obstacleX;
                    const float dy = static_cast<float>(y) - obstacleY;
                    const float dz = static_cast<float>(z) - obstacleZ;
                    const bool solid = y == 0 || y + 1 == kHeight || dx * dx + dy * dy + dz * dz <= radiusSquared;
                    const float ux = solid ? 0.0F : kInitialVelocity;
                    const auto site = (static_cast<std::size_t>(z) * kHeight + y) * kWidth + x;
                    for (std::size_t i = 0; i < kDirections; ++i)
                        populations[site * kDirections + i] = equilibrium(i, 1.0F, ux);
                }
            }
        }
        return populations;
    }

    std::string load_shader(const char* path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error(std::string("Unable to open shader: ") + path);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    template <typename Handle>
    void require_handle(Handle handle, const char* resource)
    {
        if (!handle)
            throw std::runtime_error(std::string("Failed to create WebGPU ") + resource);
    }

    // Blocks until every command already submitted to `queue` has finished on the GPU.
    void wait_for_queue([[maybe_unused]] WGPUDevice device, WGPUQueue queue)
    {
        WGPUQueueWorkDoneStatus completion = WGPUQueueWorkDoneStatus_Unknown;
        WGPUQueueWorkDoneCallbackInfo callbackInfo{};
        callbackInfo.userdata1 = &completion;
        callbackInfo.callback = [](WGPUQueueWorkDoneStatus status, void* userdata1, void*) {
            *static_cast<WGPUQueueWorkDoneStatus*>(userdata1) = status;
        };

#if defined(WEBGPU_BACKEND_WGPU)
        // wgpu-native (v24.0.0.2) does not implement wgpuInstanceWaitAny: calling it
        // aborts the process with a "not implemented" panic. Its native extension
        // wgpuDevicePoll(wait = true) blocks until all submitted work has completed
        // and fires the pending work-done callbacks before returning.
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents;
        wgpuQueueOnSubmittedWorkDone(queue, callbackInfo);
        wgpuDevicePoll(device, /*wait=*/true, nullptr);
#else
        callbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
        WGPUFutureWaitInfo waitInfo{};
        waitInfo.future = wgpuQueueOnSubmittedWorkDone(queue, callbackInfo);
        const auto waitStatus = wgpuInstanceWaitAny(ame::gpu::Instance::get(), 1, &waitInfo, std::numeric_limits<std::uint64_t>::max());
        if (waitStatus != WGPUWaitStatus_Success || !waitInfo.completed)
            throw std::runtime_error("wgpuInstanceWaitAny failed while waiting for the LBM submission");
#endif

        if (completion != WGPUQueueWorkDoneStatus_Success)
            throw std::runtime_error("WebGPU LBM submission did not complete successfully");
    }
}

void run_lbm_simulation(WGPUDevice device, WGPUAdapter adapter, WGPUSurface surface, GLFWwindow* window)
{
    if (!device || !adapter || !surface || !window)
        throw std::invalid_argument("run_lbm_simulation requires a device, adapter, surface and window");

    const auto cellCount = static_cast<std::size_t>(kWidth) * kHeight * kDepth;
    const auto populationCount = cellCount * kDirections;
    const auto populationBytes = static_cast<std::uint64_t>(populationCount * sizeof(float));
    const auto initial = make_initial_populations();
    const auto shaderCode = load_shader(LBM_SHADER_PATH);
    const auto visualShaderCode = load_shader(LBM_VISUAL_SHADER_PATH);

    WgpuHandle<WGPUQueue, wgpuQueueRelease> queue{wgpuDeviceGetQueue(device)};
    require_handle(queue.get(), "queue");

    WGPUBufferDescriptor populationDescriptor{};
    populationDescriptor.size = populationBytes;
    populationDescriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WgpuHandle<WGPUBuffer, wgpuBufferRelease> populationsA{wgpuDeviceCreateBuffer(device, &populationDescriptor)};
    WgpuHandle<WGPUBuffer, wgpuBufferRelease> populationsB{wgpuDeviceCreateBuffer(device, &populationDescriptor)};
    require_handle(populationsA.get(), "input population buffer");
    require_handle(populationsB.get(), "output population buffer");

    // Both sides start initialized so every ping-pong source is deterministic.
    wgpuQueueWriteBuffer(queue.get(), populationsA.get(), 0, initial.data(), populationBytes);
    wgpuQueueWriteBuffer(queue.get(), populationsB.get(), 0, initial.data(), populationBytes);

    const Parameters parameters{{kWidth, kHeight, kDepth, 0}, kRelaxation, kInitialVelocity,
                                static_cast<float>(kHeight) * 0.15F, 0.0F};
    WGPUBufferDescriptor parameterDescriptor{};
    parameterDescriptor.size = sizeof(parameters);
    parameterDescriptor.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    WgpuHandle<WGPUBuffer, wgpuBufferRelease> parameterBuffer{wgpuDeviceCreateBuffer(device, &parameterDescriptor)};
    require_handle(parameterBuffer.get(), "parameter buffer");
    wgpuQueueWriteBuffer(queue.get(), parameterBuffer.get(), 0, &parameters, sizeof(parameters));

    WGPUShaderSourceWGSL shaderSource{};
    shaderSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    shaderSource.code = WGPUStringView{shaderCode.data(), shaderCode.size()};
    WGPUShaderModuleDescriptor shaderDescriptor{};
    shaderDescriptor.nextInChain = &shaderSource.chain;
    WgpuHandle<WGPUShaderModule, wgpuShaderModuleRelease> shader{wgpuDeviceCreateShaderModule(device, &shaderDescriptor)};
    require_handle(shader.get(), "LBM shader module");

    std::array<WGPUBindGroupLayoutEntry, 3> layoutEntries{};
    for (std::uint32_t i = 0; i < layoutEntries.size(); ++i)
    {
        layoutEntries[i].binding = i;
        layoutEntries[i].visibility = WGPUShaderStage_Compute;
    }
    layoutEntries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    layoutEntries[0].buffer.minBindingSize = populationBytes;
    layoutEntries[1].buffer.type = WGPUBufferBindingType_Storage;
    layoutEntries[1].buffer.minBindingSize = populationBytes;
    layoutEntries[2].buffer.type = WGPUBufferBindingType_Uniform;
    layoutEntries[2].buffer.minBindingSize = sizeof(parameters);

    WGPUBindGroupLayoutDescriptor bindLayoutDescriptor{};
    bindLayoutDescriptor.entryCount = layoutEntries.size();
    bindLayoutDescriptor.entries = layoutEntries.data();
    WgpuHandle<WGPUBindGroupLayout, wgpuBindGroupLayoutRelease> bindLayout{
        wgpuDeviceCreateBindGroupLayout(device, &bindLayoutDescriptor)};
    require_handle(bindLayout.get(), "bind group layout");

    const WGPUBindGroupLayout bindLayouts[] = {bindLayout.get()};
    WGPUPipelineLayoutDescriptor pipelineLayoutDescriptor{};
    pipelineLayoutDescriptor.bindGroupLayoutCount = 1;
    pipelineLayoutDescriptor.bindGroupLayouts = bindLayouts;
    WgpuHandle<WGPUPipelineLayout, wgpuPipelineLayoutRelease> pipelineLayout{
        wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDescriptor)};
    require_handle(pipelineLayout.get(), "pipeline layout");

    WGPUProgrammableStageDescriptor computeStage{};
    computeStage.module = shader.get();
    computeStage.entryPoint = WGPUStringView{"simulate", 8};
    WGPUComputePipelineDescriptor pipelineDescriptor{};
    pipelineDescriptor.layout = pipelineLayout.get();
    pipelineDescriptor.compute = computeStage;
    WgpuHandle<WGPUComputePipeline, wgpuComputePipelineRelease> pipeline{
        wgpuDeviceCreateComputePipeline(device, &pipelineDescriptor)};
    require_handle(pipeline.get(), "compute pipeline");

    const auto create_bind_group = [&](WGPUBuffer source, WGPUBuffer destination) {
        std::array<WGPUBindGroupEntry, 3> entries{};
        entries[0].binding = 0;
        entries[0].buffer = source;
        entries[0].size = populationBytes;
        entries[1].binding = 1;
        entries[1].buffer = destination;
        entries[1].size = populationBytes;
        entries[2].binding = 2;
        entries[2].buffer = parameterBuffer.get();
        entries[2].size = sizeof(parameters);
        WGPUBindGroupDescriptor descriptor{};
        descriptor.layout = bindLayout.get();
        descriptor.entryCount = entries.size();
        descriptor.entries = entries.data();
        WgpuHandle<WGPUBindGroup, wgpuBindGroupRelease> group{wgpuDeviceCreateBindGroup(device, &descriptor)};
        require_handle(group.get(), "bind group");
        return group;
    };
    auto bindGroupAB = create_bind_group(populationsA.get(), populationsB.get());
    auto bindGroupBA = create_bind_group(populationsB.get(), populationsA.get());

    // The render pipeline binds the current ping-pong population buffer as
    // read-only storage. Compute and rendering are encoded in the same command
    // buffer, so WebGPU inserts the required storage-buffer transition.
    std::array<WGPUBindGroupLayoutEntry, 2> renderLayoutEntries{};
    renderLayoutEntries[0].binding = 0;
    renderLayoutEntries[0].visibility = WGPUShaderStage_Fragment;
    renderLayoutEntries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    renderLayoutEntries[0].buffer.minBindingSize = populationBytes;
    renderLayoutEntries[1].binding = 1;
    renderLayoutEntries[1].visibility = WGPUShaderStage_Fragment;
    renderLayoutEntries[1].buffer.type = WGPUBufferBindingType_Uniform;
    renderLayoutEntries[1].buffer.minBindingSize = sizeof(parameters);
    WGPUBindGroupLayoutDescriptor renderLayoutDesc{};
    renderLayoutDesc.entryCount = renderLayoutEntries.size();
    renderLayoutDesc.entries = renderLayoutEntries.data();
    WgpuHandle<WGPUBindGroupLayout, wgpuBindGroupLayoutRelease> renderLayout{
        wgpuDeviceCreateBindGroupLayout(device, &renderLayoutDesc)};
    require_handle(renderLayout.get(), "visualization bind group layout");

    const auto make_render_group = [&](WGPUBuffer populations) {
        std::array<WGPUBindGroupEntry, 2> entries{};
        entries[0].binding = 0; entries[0].buffer = populations; entries[0].size = populationBytes;
        entries[1].binding = 1; entries[1].buffer = parameterBuffer.get(); entries[1].size = sizeof(parameters);
        WGPUBindGroupDescriptor desc{};
        desc.layout = renderLayout.get(); desc.entryCount = entries.size(); desc.entries = entries.data();
        WgpuHandle<WGPUBindGroup, wgpuBindGroupRelease> result{wgpuDeviceCreateBindGroup(device, &desc)};
        require_handle(result.get(), "visualization bind group");
        return result;
    };
    auto renderGroupA = make_render_group(populationsA.get());
    auto renderGroupB = make_render_group(populationsB.get());

    WGPUShaderSourceWGSL visualSource{};
    visualSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    visualSource.code = WGPUStringView{visualShaderCode.data(), visualShaderCode.size()};
    WGPUShaderModuleDescriptor visualShaderDesc{};
    visualShaderDesc.nextInChain = &visualSource.chain;
    WgpuHandle<WGPUShaderModule, wgpuShaderModuleRelease> visualShader{
        wgpuDeviceCreateShaderModule(device, &visualShaderDesc)};
    require_handle(visualShader.get(), "visualization shader");

    WGPUSurfaceCapabilities capabilities{};
    wgpuSurfaceGetCapabilities(surface, adapter, &capabilities);
    if (capabilities.formatCount == 0)
        throw std::runtime_error("Surface exposes no texture format");
    const WGPUTextureFormat surfaceFormat = capabilities.formats[0];
    wgpuSurfaceCapabilitiesFreeMembers(capabilities);

    const WGPUBindGroupLayout renderLayouts[] = {renderLayout.get()};
    WGPUPipelineLayoutDescriptor renderPipelineLayoutDesc{};
    renderPipelineLayoutDesc.bindGroupLayoutCount = 1;
    renderPipelineLayoutDesc.bindGroupLayouts = renderLayouts;
    WgpuHandle<WGPUPipelineLayout, wgpuPipelineLayoutRelease> renderPipelineLayout{
        wgpuDeviceCreatePipelineLayout(device, &renderPipelineLayoutDesc)};
    WGPUColorTargetState colorTarget{};
    colorTarget.format = surfaceFormat;
    colorTarget.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fragment{};
    fragment.module = visualShader.get();
    fragment.entryPoint = WGPUStringView{"fragment_main", 13};
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;
    WGPURenderPipelineDescriptor renderPipelineDesc{};
    renderPipelineDesc.layout = renderPipelineLayout.get();
    renderPipelineDesc.vertex.module = visualShader.get();
    renderPipelineDesc.vertex.entryPoint = WGPUStringView{"vertex_main", 11};
    renderPipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    renderPipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
    renderPipelineDesc.primitive.cullMode = WGPUCullMode_None;
    renderPipelineDesc.multisample.count = 1;
    renderPipelineDesc.multisample.mask = ~0u;
    renderPipelineDesc.fragment = &fragment;
    WgpuHandle<WGPURenderPipeline, wgpuRenderPipelineRelease> renderPipeline{
        wgpuDeviceCreateRenderPipeline(device, &renderPipelineDesc)};
    require_handle(renderPipeline.get(), "visualization render pipeline");

    int configuredWidth = 0;
    int configuredHeight = 0;
    std::uint32_t step = 0;
    bool completionReported = false;
    const auto groupsX = (kWidth + kWorkgroupSize - 1) / kWorkgroupSize;
    const auto groupsY = (kHeight + kWorkgroupSize - 1) / kWorkgroupSize;
    const auto groupsZ = (kDepth + kWorkgroupSize - 1) / kWorkgroupSize;
    std::cout << "LBM D3Q19 real-time GPU visualization (close the window to stop).\n";

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width <= 0 || height <= 0)
            continue;
        if (width != configuredWidth || height != configuredHeight)
        {
            WGPUSurfaceConfiguration config{};
            config.device = device;
            config.format = surfaceFormat;
            config.usage = WGPUTextureUsage_RenderAttachment;
            config.width = static_cast<std::uint32_t>(width);
            config.height = static_cast<std::uint32_t>(height);
            config.presentMode = WGPUPresentMode_Fifo;
            config.alphaMode = WGPUCompositeAlphaMode_Auto;
            wgpuSurfaceConfigure(surface, &config);
            configuredWidth = width;
            configuredHeight = height;
        }

        WGPUSurfaceTexture surfaceTexture{};
        wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
        if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
            continue;
        WGPUTextureViewDescriptor viewDesc{};
        WgpuHandle<WGPUTextureView, wgpuTextureViewRelease> view{
            wgpuTextureCreateView(surfaceTexture.texture, &viewDesc)};
        wgpuTextureRelease(surfaceTexture.texture);
        require_handle(view.get(), "surface texture view");

        WGPUCommandEncoderDescriptor encoderDesc{};
        WgpuHandle<WGPUCommandEncoder, wgpuCommandEncoderRelease> encoder{
            wgpuDeviceCreateCommandEncoder(device, &encoderDesc)};
        if (step < kSteps)
        {
            WGPUComputePassDescriptor computeDesc{};
            WGPUComputePassEncoder compute = wgpuCommandEncoderBeginComputePass(encoder.get(), &computeDesc);
            require_handle(compute, "compute pass");
            wgpuComputePassEncoderSetPipeline(compute, pipeline.get());
            wgpuComputePassEncoderSetBindGroup(compute, 0, (step % 2 == 0) ? bindGroupAB.get() : bindGroupBA.get(), 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(compute, groupsX, groupsY, groupsZ);
            wgpuComputePassEncoderEnd(compute);
            wgpuComputePassEncoderRelease(compute);
        }

        WGPURenderPassColorAttachment attachment{};
        attachment.view = view.get();
        attachment.loadOp = WGPULoadOp_Clear;
        attachment.storeOp = WGPUStoreOp_Store;
        attachment.clearValue = WGPUColor{0.0, 0.0, 0.0, 1.0};
        WGPURenderPassDescriptor renderDesc{};
        renderDesc.colorAttachmentCount = 1;
        renderDesc.colorAttachments = &attachment;
        WGPURenderPassEncoder render = wgpuCommandEncoderBeginRenderPass(encoder.get(), &renderDesc);
        wgpuRenderPassEncoderSetPipeline(render, renderPipeline.get());
        // During simulation, even steps write B and odd steps write A. Once
        // all steps are done, keep presenting the final buffer until close.
        const bool currentIsB = (step < kSteps) ? (step % 2 == 0) : (kSteps % 2 == 1);
        wgpuRenderPassEncoderSetBindGroup(render, 0, currentIsB ? renderGroupB.get() : renderGroupA.get(), 0, nullptr);
        wgpuRenderPassEncoderDraw(render, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(render);
        wgpuRenderPassEncoderRelease(render);

        WgpuHandle<WGPUCommandBuffer, wgpuCommandBufferRelease> commands{
            wgpuCommandEncoderFinish(encoder.get(), nullptr)};
        const WGPUCommandBuffer submitted = commands.get();
        wgpuQueueSubmit(queue.get(), 1, &submitted);
        wgpuSurfacePresent(surface);
#if defined(WEBGPU_BACKEND_WGPU)
        wgpuDevicePoll(device, false, nullptr);
#endif
        if (step < kSteps)
            ++step;
        if (step == kSteps && !completionReported)
        {
            std::cout << "LBM D3Q19: " << kWidth << 'x' << kHeight << 'x' << kDepth
                      << ", " << kSteps << " GPU iterations submitted; displaying final field.\n" << std::flush;
            completionReported = true;
        }
    }
    wait_for_queue(device, queue.get());
    wgpuSurfaceUnconfigure(surface);
}
