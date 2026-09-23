#pragma once

#include <webgpu/webgpu.h>

// Run the D3Q19 compute simulation on an already-created WebGPU device.
void run_lbm_simulation(WGPUDevice device);
