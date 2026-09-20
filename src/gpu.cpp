#include <gpu.h>

#include <iostream>
#include <future>

namespace ame::gpu
{
    //////////////////////////////////////////////
    // Adapter class implementation

    Adapter::Adapter(::WGPUAdapter adapter) : adapter(adapter)
    {
        std::cout << "WebGPU adapter created: " << adapter << std::endl;
    }

    Adapter::Adapter(Adapter &other) : adapter(other.adapter)
    {
        std::cout << "WebGPU adapter copied: " << adapter << std::endl;
    }

    Adapter::Adapter(Adapter &&other) noexcept : adapter(other.adapter)
    {
        std::cout << "WebGPU adapter moved: " << adapter << std::endl;
        other.adapter = nullptr;
    }

    Adapter::~Adapter()
    {
        if (adapter)
        {
            ::wgpuAdapterRelease(adapter);
            std::cout << "WebGPU adapter released: " << adapter << std::endl;
            adapter = nullptr;
        }
    }

    ::WGPUAdapter Adapter::get() const
    {
        return adapter;
    }
    std::future<Device> Adapter::request_device_async(::WGPUDeviceDescriptor const &descriptor)
    {
        if (promise.get_future().valid())
            promise.set_exception(std::make_exception_ptr(std::runtime_error("Request device called while a previous request is still pending")));
        promise = std::promise<Device>(); // Reinit promise with a not set value
        auto future = promise.get_future();
        wgpuAdapterRequestDevice(
            adapter,
            &descriptor,
            { // WGPURequestDeviceCallbackInfo
                nullptr,                      // WGPUChainedStruct const * nextInChain;
                WGPUCallbackMode_WaitAnyOnly, // WGPUCallbackMode mode;
                [](::WGPURequestDeviceStatus status, ::WGPUDevice device, WGPUStringView message, void *userdata1, void *)
                {
                    auto promise = static_cast<std::promise<Device> *>(userdata1);
                    if (status == WGPURequestDeviceStatus_Success)
                    {
                        promise->set_value(Device(device));
                    }
                    else
                    {
                        promise->set_exception(std::make_exception_ptr(std::runtime_error(message.data ? std::string(message.data, message.length) : "Unknown error")));
                    }
                },
                static_cast<void *>(&promise), // userdata1
                nullptr                        // userdata2
            });

        return future;
    }

    //////////////////////////////////////////////
    // Device class implementation
    Device::Device(::WGPUDevice device) : device(device) {}

    Device::Device(Device &other) : device(other.device)
    {
        std::cout << "WebGPU device copied: " << device << std::endl;
    }

    Device::Device(Device &&other) noexcept : device(other.device)
    {
        std::cout << "WebGPU device moved: " << device << std::endl;
        other.device = nullptr;
    }

    ::WGPUDevice Device::get() const
    {
        return device;
    }



    Device::~Device()
    {
        if (device)
        {
            wgpuDeviceRelease(device);
            std::cout << "WebGPU device released: " << device << std::endl;
            device = nullptr;
        }
    }

    //////////////////////////////////////////////
    // Instance class implementation
    Instance::Instance()
    {
        create_instance();
        std::cout << "WebGPU instance created: " << instance << std::endl;
    }
    Instance::~Instance()
    {
        if (instance)
        {
            wgpuInstanceRelease(instance);
            std::cout << "WebGPU instance released: " << instance << std::endl;
            instance = nullptr;
        }
    }
    ::WGPUInstance Instance::get()
    {
        return get_singleton().instance;
    }

    Instance &Instance::get_singleton()
    {
        if (!singleton)
        {
            singleton = std::make_unique<Instance>();
        }
        return *singleton;
    }
    void Instance::create_instance()
    {
#ifdef WEBGPU_BACKEND_EMSCRIPTEN
        instance = wgpuCreateInstance(nullptr);
#else  //  WEBGPU_BACKEND_EMSCRIPTEN
        WGPUInstanceDescriptor desc = {};
        desc.nextInChain = nullptr;
        instance = wgpuCreateInstance(&desc);
#endif //  WEBGPU_BACKEND_EMSCRIPTEN
    }

    std::future<Adapter> Instance::request_adapter_async(::WGPURequestAdapterOptions const &options)
    {
        if (promise.get_future().valid())
            promise.set_exception(std::make_exception_ptr(std::runtime_error("Request adapter called while a previous request is still pending")));
        promise = std::promise<Adapter>(); // Reinit promise with a not set value
        auto future = promise.get_future();

        WGPURequestAdapterCallbackInfo callbackInfo = {};
        callbackInfo.nextInChain = nullptr;
        callbackInfo.userdata1 = &promise;
        callbackInfo.userdata2 = nullptr;
        callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void *userdata1, void *)
        {
            auto promise = static_cast<std::promise<Adapter> *>(userdata1);
            if (status == WGPURequestAdapterStatus_Success)
            {
                promise->set_value(Adapter(adapter));
            }
            else
            {
                promise->set_exception(std::make_exception_ptr(std::runtime_error(message.data != nullptr ? message.data : "Unknown error")));
                std::cerr << "Failed to request adapter: " << (message.data != nullptr ? message.data : "Unknown error") << std::endl;
            }
        };
        wgpuInstanceRequestAdapter(get(), &options, callbackInfo);
        return future;
    }
    std::unique_ptr<ame::gpu::Instance> ame::gpu::Instance::singleton = nullptr;
}
