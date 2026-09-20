#pragma once
#include <webgpu/webgpu.h>
#include <memory>
#include <future>

namespace ame::gpu
{
    class Device
    {
    public:
        Device(::WGPUDevice device);
        Device(Device & other);
        Device(Device && other) noexcept;
        ~Device();
        ::WGPUDevice get() const;
        operator ::WGPUDevice() const { return device; }
        operator bool() const { return device != nullptr; }
    private:
        ::WGPUDevice device = nullptr;
    };

    class Adapter
    {
    public:
        Adapter(::WGPUAdapter adapter);
        Adapter(Adapter & other);
        Adapter(Adapter && other) noexcept;
        ~Adapter();
        operator ::WGPUAdapter() const { return adapter; }
        operator bool() const { return adapter != nullptr; }
        ::WGPUAdapter get() const;
        std::future<Device> request_device_async(::WGPUDeviceDescriptor const &descriptor);
        Device request_device(::WGPUDeviceDescriptor const &descriptor) { return request_device_async(descriptor).get(); }

    private:
        ::WGPUAdapter adapter = nullptr;
        std::promise<Device> promise;
    };


    class Instance
    {
    public:
        Instance();
        ~Instance();
        // Deleted copy constructor and assignment operator to prevent copying
        Instance(const Instance &) = delete;
        Instance &operator=(const Instance &) = delete;
        static ::WGPUInstance get();
        static Instance &get_singleton();
        static Adapter Request_Adapter(::WGPURequestAdapterOptions const &options) 
        { return get_singleton().request_adapter(options); }
        static std::future<Adapter> Request_Adapter_Async(::WGPURequestAdapterOptions const &options) 
        { return get_singleton().request_adapter_async(options); }

    private:
        std::promise<Adapter> promise;
        ::WGPUInstance instance = nullptr;
        void create_instance();
        std::future<Adapter> request_adapter_async(::WGPURequestAdapterOptions const &options);
        Adapter request_adapter(::WGPURequestAdapterOptions const &options) 
        { return request_adapter_async(options).get(); }
        // static
        static std::unique_ptr<Instance> singleton;
    };
}
