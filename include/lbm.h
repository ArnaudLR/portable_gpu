#pragma once

#include <webgpu/webgpu.h>

struct GLFWwindow;

// Simulate and render one GPU-only D3Q19 frame per iteration until the window closes.
void run_lbm_simulation(WGPUDevice device, WGPUAdapter adapter, WGPUSurface surface, GLFWwindow* window);
