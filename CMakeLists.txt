cmake_minimum_required(VERSION 3.0...3.25)

project(
    LearnWebGPU # name of the project, which will also be the name of the visual studio solution if you use it
    VERSION 0.1.0 # any version number
    LANGUAGES CXX C # programming languages used by the project
)
add_subdirectory(WebGPU-distribution-wgpu-v24.0.0.2)

include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
FetchContent_MakeAvailable(glfw)
FetchContent_Declare(glfw3webgpu GIT_REPOSITORY https://github.com/eliemichel/glfw3webgpu.git GIT_TAG v1.3.0-alpha)
FetchContent_MakeAvailable(glfw3webgpu)

add_executable(App 
    src/main.cpp 
    src/gpu.cpp
    src/lbm.cpp
    src/inspector.cpp
)
target_include_directories(App PRIVATE include)
target_compile_definitions(App PRIVATE
    LBM_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/lbm_kernels.wgsl"
    LBM_VISUAL_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/visualization.wgsl"
    LBM_RENDER_SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/visualization_render.wgsl"
)
# Add the 'webgpu' target as a dependency of our App
target_link_libraries(App PRIVATE webgpu glfw glfw3webgpu)
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