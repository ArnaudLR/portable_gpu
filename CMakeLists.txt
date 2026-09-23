cmake_minimum_required(VERSION 3.0...3.25)

project(
    LearnWebGPU # name of the project, which will also be the name of the visual studio solution if you use it
    VERSION 0.1.0 # any version number
    LANGUAGES CXX C # programming languages used by the project
)
add_subdirectory(WebGPU-distribution-wgpu-v24.0.0.2)

add_executable(App 
    src/main.cpp 
    src/gpu.cpp
    src/lbm.cpp
)
target_include_directories(App PRIVATE include)
target_compile_definitions(App PRIVATE LBM_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/lbm_kernels.wgsl")
# Add the 'webgpu' target as a dependency of our App
target_link_libraries(App PRIVATE webgpu)
# The application's binary must find wgpu.dll or libwgpu.so at runtime,
# so we automatically copy it (it's called WGPU_RUNTIME_LIB in general)
# next to the binary.
target_copy_webgpu_binaries(App)

set_target_properties(App PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    COMPILE_WARNING_AS_ERROR ON
)

if (MSVC)
    target_compile_options(App PRIVATE /W4)
else()
    target_compile_options(App PRIVATE -Wall -Wextra -pedantic)
endif()

if (XCODE)
    set_target_properties(App PROPERTIES
        XCODE_GENERATE_SCHEME ON
        XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE "Metal"
    )
endif()

# Options that are specific to Emscripten
if (EMSCRIPTEN)
	# Generate a full web page rather than a simple WebAssembly module
	set_target_properties(App PROPERTIES SUFFIX ".html")
endif()