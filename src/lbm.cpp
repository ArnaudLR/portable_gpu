#include "lbm.h"
#include <gpu.h>

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

    std::string load_shader()
    {
        std::ifstream input(LBM_SHADER_PATH, std::ios::binary);
        if (!input)
            throw std::runtime_error(std::string("Unable to open LBM shader: ") + LBM_SHADER_PATH);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    template <typename Handle>
    void require_handle(Handle handle, const char* resource)
    {
        if (!handle)
            throw std::runtime_error(std::string("Failed to create WebGPU ") + resource);
    }

    void wait_for_queue(WGPUQueue queue)
    {
        WGPUQueueWorkDoneStatus completion = WGPUQueueWorkDoneStatus_Unknown;
        WGPUQueueWorkDoneCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
        callbackInfo.userdata1 = &completion;
        callbackInfo.callback = [](WGPUQueueWorkDoneStatus status, void* userdata1, void*) {
            *static_cast<WGPUQueueWorkDoneStatus*>(userdata1) = status;
        };

        const WGPUFuture future = wgpuQueueOnSubmittedWorkDone(queue, callbackInfo);
        WGPUFutureWaitInfo waitInfo{};
        waitInfo.future = future;
        const auto waitStatus = wgpuInstanceWaitAny(ame::gpu::Instance::get(), 1, &waitInfo, std::numeric_limits<std::uint64_t>::max());
        if (waitStatus != WGPUWaitStatus_Success || !waitInfo.completed ||
            completion != WGPUQueueWorkDoneStatus_Success)
            throw std::runtime_error("WebGPU LBM submission did not complete successfully");
    }
}

void run_lbm_simulation(WGPUDevice device)
{
    if (!device)
        throw std::invalid_argument("run_lbm_simulation requires a valid WebGPU device");

    const auto cellCount = static_cast<std::size_t>(kWidth) * kHeight * kDepth;
    const auto populationCount = cellCount * kDirections;
    const auto populationBytes = static_cast<WGPUBufferSize>(populationCount * sizeof(float));
    const auto initial = make_initial_populations();
    const auto shaderCode = load_shader();

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

    WGPUCommandEncoderDescriptor encoderDescriptor{};
    WgpuHandle<WGPUCommandEncoder, wgpuCommandEncoderRelease> encoder{
        wgpuDeviceCreateCommandEncoder(device, &encoderDescriptor)};
    require_handle(encoder.get(), "command encoder");

    WGPUComputePassDescriptor passDescriptor{};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder.get(), &passDescriptor);
    require_handle(pass, "compute pass");
    wgpuComputePassEncoderSetPipeline(pass, pipeline.get());
    const auto groupsX = (kWidth + kWorkgroupSize - 1) / kWorkgroupSize;
    const auto groupsY = (kHeight + kWorkgroupSize - 1) / kWorkgroupSize;
    const auto groupsZ = (kDepth + kWorkgroupSize - 1) / kWorkgroupSize;
    for (std::uint32_t step = 0; step < kSteps; ++step)
    {
        const auto group = (step % 2 == 0) ? bindGroupAB.get() : bindGroupBA.get();
        wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, groupsX, groupsY, groupsZ);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WgpuHandle<WGPUCommandBuffer, wgpuCommandBufferRelease> commandBuffer{
        wgpuCommandEncoderFinish(encoder.get(), nullptr)};
    require_handle(commandBuffer.get(), "command buffer");
    const WGPUCommandBuffer submittedCommand = commandBuffer.get();
    wgpuQueueSubmit(queue.get(), 1, &submittedCommand);
    wait_for_queue(queue.get());

    std::cout << "LBM D3Q19: " << kWidth << 'x' << kHeight << 'x' << kDepth
              << ", " << kSteps << " GPU steps submitted (BGK, bounce-back walls and sphere).\n";
}
