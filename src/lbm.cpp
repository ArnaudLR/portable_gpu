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
#ifndef LBM_RENDER_SHADER_PATH
#define LBM_RENDER_SHADER_PATH "src/visualization_render.wgsl"
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
    const auto renderShaderCode = load_shader(LBM_RENDER_SHADER_PATH);

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

    // Convert the population buffer to a small GPU texture in a compute pass.
    // The graphics pipeline only samples that texture; this avoids compiling
    // storage-buffer ray marching in the fragment stage on older Intel drivers.
    WGPUTextureDescriptor projectionDesc{};
    projectionDesc.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
    projectionDesc.dimension = WGPUTextureDimension_2D;
    projectionDesc.size = WGPUExtent3D{kWidth, kHeight, 1};
    projectionDesc.format = WGPUTextureFormat_RGBA8Unorm;
    projectionDesc.mipLevelCount = 1;
    projectionDesc.sampleCount = 1;
    WgpuHandle<WGPUTexture, wgpuTextureRelease> projection{
        wgpuDeviceCreateTexture(device, &projectionDesc)};
    require_handle(projection.get(), "projection texture");
    WGPUTextureViewDescriptor projectionViewDesc{};
    WgpuHandle<WGPUTextureView, wgpuTextureViewRelease> projectionView{
        wgpuTextureCreateView(projection.get(), &projectionViewDesc)};
    require_handle(projectionView.get(), "projection texture view");

    WGPUShaderSourceWGSL visualSource{};
    visualSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    visualSource.code = WGPUStringView{visualShaderCode.data(), visualShaderCode.size()};
    WGPUShaderModuleDescriptor visualShaderDesc{};
    visualShaderDesc.nextInChain = &visualSource.chain;
    WgpuHandle<WGPUShaderModule, wgpuShaderModuleRelease> visualShader{
        wgpuDeviceCreateShaderModule(device, &visualShaderDesc)};
    require_handle(visualShader.get(), "visualization shader");
    WGPUShaderSourceWGSL renderSource{};
    renderSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    renderSource.code = WGPUStringView{renderShaderCode.data(), renderShaderCode.size()};
    WGPUShaderModuleDescriptor renderShaderDesc{};
    renderShaderDesc.nextInChain = &renderSource.chain;
    WgpuHandle<WGPUShaderModule, wgpuShaderModuleRelease> renderShader{
        wgpuDeviceCreateShaderModule(device, &renderShaderDesc)};
    require_handle(renderShader.get(), "presentation shader");

    std::array<WGPUBindGroupLayoutEntry, 3> projectEntries{};
    projectEntries[0].binding=0; projectEntries[0].visibility=WGPUShaderStage_Compute;
    projectEntries[0].buffer.type=WGPUBufferBindingType_ReadOnlyStorage; projectEntries[0].buffer.minBindingSize=populationBytes;
    projectEntries[1].binding=1; projectEntries[1].visibility=WGPUShaderStage_Compute;
    projectEntries[1].buffer.type=WGPUBufferBindingType_Uniform; projectEntries[1].buffer.minBindingSize=sizeof(parameters);
    projectEntries[2].binding=2; projectEntries[2].visibility=WGPUShaderStage_Compute;
    projectEntries[2].storageTexture.access=WGPUStorageTextureAccess_WriteOnly;
    projectEntries[2].storageTexture.format=WGPUTextureFormat_RGBA8Unorm;
    projectEntries[2].storageTexture.viewDimension=WGPUTextureViewDimension_2D;
    WGPUBindGroupLayoutDescriptor projectLayoutDesc{};
    projectLayoutDesc.entryCount=projectEntries.size(); projectLayoutDesc.entries=projectEntries.data();
    WgpuHandle<WGPUBindGroupLayout, wgpuBindGroupLayoutRelease> projectLayout{
        wgpuDeviceCreateBindGroupLayout(device,&projectLayoutDesc)};
    const WGPUBindGroupLayout projectLayouts[]={projectLayout.get()};
    WGPUPipelineLayoutDescriptor projectPipelineLayoutDesc{};
    projectPipelineLayoutDesc.bindGroupLayoutCount=1; projectPipelineLayoutDesc.bindGroupLayouts=projectLayouts;
    WgpuHandle<WGPUPipelineLayout, wgpuPipelineLayoutRelease> projectPipelineLayout{
        wgpuDeviceCreatePipelineLayout(device,&projectPipelineLayoutDesc)};
    WGPUComputePipelineDescriptor projectPipelineDesc{};
    projectPipelineDesc.layout=projectPipelineLayout.get(); projectPipelineDesc.compute.module=visualShader.get();
    projectPipelineDesc.compute.entryPoint=WGPUStringView{"project",7};
    WgpuHandle<WGPUComputePipeline, wgpuComputePipelineRelease> projectPipeline{
        wgpuDeviceCreateComputePipeline(device,&projectPipelineDesc)};
    require_handle(projectPipeline.get(), "projection compute pipeline");
    const auto makeProjectGroup=[&](WGPUBuffer buffer){
        std::array<WGPUBindGroupEntry,3> e{};
        e[0].binding=0;e[0].buffer=buffer;e[0].size=populationBytes;
        e[1].binding=1;e[1].buffer=parameterBuffer.get();e[1].size=sizeof(parameters);
        e[2].binding=2;e[2].textureView=projectionView.get();
        WGPUBindGroupDescriptor d{};d.layout=projectLayout.get();d.entryCount=e.size();d.entries=e.data();
        WgpuHandle<WGPUBindGroup,wgpuBindGroupRelease> g{wgpuDeviceCreateBindGroup(device,&d)};
        require_handle(g.get(),"projection bind group");return g;};
    auto projectGroupA=makeProjectGroup(populationsA.get()); auto projectGroupB=makeProjectGroup(populationsB.get());

    std::array<WGPUBindGroupLayoutEntry,2> textureEntries{};
    textureEntries[0].binding=0;textureEntries[0].visibility=WGPUShaderStage_Fragment;
    textureEntries[0].texture.sampleType=WGPUTextureSampleType_Float;textureEntries[0].texture.viewDimension=WGPUTextureViewDimension_2D;
    textureEntries[1].binding=1;textureEntries[1].visibility=WGPUShaderStage_Fragment;
    textureEntries[1].sampler.type=WGPUSamplerBindingType_Filtering;
    WGPUBindGroupLayoutDescriptor textureLayoutDesc{};textureLayoutDesc.entryCount=textureEntries.size();textureLayoutDesc.entries=textureEntries.data();
    WgpuHandle<WGPUBindGroupLayout,wgpuBindGroupLayoutRelease> textureLayout{wgpuDeviceCreateBindGroupLayout(device,&textureLayoutDesc)};
    WGPUSamplerDescriptor samplerDesc{};samplerDesc.magFilter=WGPUFilterMode_Linear;samplerDesc.minFilter=WGPUFilterMode_Linear;
    WgpuHandle<WGPUSampler,wgpuSamplerRelease> sampler{wgpuDeviceCreateSampler(device,&samplerDesc)};
    std::array<WGPUBindGroupEntry,2> textureGroupEntries{};
    textureGroupEntries[0].binding=0;textureGroupEntries[0].textureView=projectionView.get();
    textureGroupEntries[1].binding=1;textureGroupEntries[1].sampler=sampler.get();
    WGPUBindGroupDescriptor textureGroupDesc{};textureGroupDesc.layout=textureLayout.get();textureGroupDesc.entryCount=2;textureGroupDesc.entries=textureGroupEntries.data();
    WgpuHandle<WGPUBindGroup,wgpuBindGroupRelease> textureGroup{wgpuDeviceCreateBindGroup(device,&textureGroupDesc)};

    WGPUSurfaceCapabilities capabilities{}; wgpuSurfaceGetCapabilities(surface,adapter,&capabilities);
    if(capabilities.formatCount==0) throw std::runtime_error("Surface exposes no texture format");
    const WGPUTextureFormat surfaceFormat=capabilities.formats[0]; wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    const WGPUBindGroupLayout renderLayouts[]={textureLayout.get()};
    WGPUPipelineLayoutDescriptor renderLayoutDesc{};renderLayoutDesc.bindGroupLayoutCount=1;renderLayoutDesc.bindGroupLayouts=renderLayouts;
    WgpuHandle<WGPUPipelineLayout,wgpuPipelineLayoutRelease> renderPipelineLayout{wgpuDeviceCreatePipelineLayout(device,&renderLayoutDesc)};
    WGPUColorTargetState target{};target.format=surfaceFormat;target.writeMask=WGPUColorWriteMask_All;
    WGPUFragmentState fragment{};fragment.module=renderShader.get();fragment.entryPoint=WGPUStringView{"fragment_main",13};fragment.targetCount=1;fragment.targets=&target;
    WGPURenderPipelineDescriptor renderDesc{};renderDesc.layout=renderPipelineLayout.get();renderDesc.vertex.module=renderShader.get();
    renderDesc.vertex.entryPoint=WGPUStringView{"vertex_main",11};renderDesc.primitive.topology=WGPUPrimitiveTopology_TriangleList;
    renderDesc.primitive.frontFace=WGPUFrontFace_CCW;renderDesc.primitive.cullMode=WGPUCullMode_None;
    renderDesc.multisample.count=1;renderDesc.multisample.mask=~0u;renderDesc.fragment=&fragment;
    WgpuHandle<WGPURenderPipeline,wgpuRenderPipelineRelease> renderPipeline{wgpuDeviceCreateRenderPipeline(device,&renderDesc)};
    require_handle(renderPipeline.get(),"visualization render pipeline");

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

        WGPUComputePassDescriptor projectionPassDesc{};
        WGPUComputePassEncoder projectionPass = wgpuCommandEncoderBeginComputePass(encoder.get(), &projectionPassDesc);
        require_handle(projectionPass, "projection compute pass");
        wgpuComputePassEncoderSetPipeline(projectionPass, projectPipeline.get());
        const bool currentIsB = (step < kSteps) ? (step % 2 == 0) : (kSteps % 2 == 1);
        wgpuComputePassEncoderSetBindGroup(projectionPass, 0, currentIsB ? projectGroupB.get() : projectGroupA.get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(projectionPass, (kWidth + 7) / 8, (kHeight + 7) / 8, 1);
        wgpuComputePassEncoderEnd(projectionPass);
        wgpuComputePassEncoderRelease(projectionPass);

        WGPURenderPassColorAttachment attachment{};
        attachment.view = view.get();
        attachment.loadOp = WGPULoadOp_Clear;
        attachment.storeOp = WGPUStoreOp_Store;
        attachment.clearValue = WGPUColor{0.0, 0.0, 0.0, 1.0};
        WGPURenderPassDescriptor renderPassDesc{};
        renderPassDesc.colorAttachmentCount = 1;
        renderPassDesc.colorAttachments = &attachment;
        WGPURenderPassEncoder render = wgpuCommandEncoderBeginRenderPass(encoder.get(), &renderPassDesc);
        wgpuRenderPassEncoderSetPipeline(render, renderPipeline.get());
        wgpuRenderPassEncoderSetBindGroup(render, 0, textureGroup.get(), 0, nullptr);
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
