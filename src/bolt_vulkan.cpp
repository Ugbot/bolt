// bolt_vulkan.cpp -- implementation of bolt::vulkan's device/queue/memory
// scaffolding (BLLM-46). See bolt_vulkan.h for the full design rationale.
//
// Tiger Style adherence: no exceptions (Vulkan's C API doesn't throw
// anyway), noexcept everywhere, every failure path returns false/nullptr
// rather than asserting -- absence of Vulkan (or of a compute-capable
// device) is an expected, not exceptional, runtime condition this module
// must handle cleanly for every caller to fall back to CPU.

#include "bolt/bolt_vulkan.h"

#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

namespace bolt::vulkan {

namespace {

// Finds the first queue family index on `phys` that supports compute,
// preferring one that is DEDICATED compute (no graphics bit) if any such
// family exists, else the first family with VK_QUEUE_COMPUTE_BIT at all
// (a combined graphics+compute family, which every Vulkan-capable GPU is
// required to expose per spec if it supports compute at all).
bool find_compute_queue_family(VkPhysicalDevice phys, uint32_t* out_index) noexcept {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
    if (count == 0) return false;

    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, families.data());

    int32_t combined_index = -1;
    for (uint32_t i = 0; i < count; ++i) {
        const bool has_compute = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (!has_compute) continue;
        const bool has_graphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (!has_graphics) {
            *out_index = i;
            return true;  // dedicated compute family -- best case, stop immediately
        }
        if (combined_index < 0) combined_index = static_cast<int32_t>(i);
    }
    if (combined_index >= 0) {
        *out_index = static_cast<uint32_t>(combined_index);
        return true;
    }
    return false;
}

// Picks the "best" physical device among those exposing a compute queue
// family: prefers DISCRETE_GPU, then INTEGRATED_GPU, then anything else
// (VIRTUAL_GPU/CPU/OTHER) that still qualifies. Returns false if NONE of
// `devices` has a compute-capable queue family.
bool pick_physical_device(const std::vector<VkPhysicalDevice>& devices,
                          VkPhysicalDevice* out_device, uint32_t* out_queue_family) noexcept {
    VkPhysicalDevice best = VK_NULL_HANDLE;
    uint32_t best_queue_family = 0;
    int best_rank = -1;  // higher is better

    for (VkPhysicalDevice dev : devices) {
        uint32_t qf;
        if (!find_compute_queue_family(dev, &qf)) continue;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        int rank = 0;
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   rank = 3; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: rank = 2; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    rank = 1; break;
            default:                                     rank = 0; break;
        }
        if (rank > best_rank) {
            best_rank = rank;
            best = dev;
            best_queue_family = qf;
        }
    }

    if (best == VK_NULL_HANDLE) return false;
    *out_device = best;
    *out_queue_family = best_queue_family;
    return true;
}

// Finds a memory type index satisfying both `type_bits` (from a
// VkMemoryRequirements) and every bit in `required_props`. Returns false
// if no memory type on this device qualifies.
bool find_memory_type(const VkPhysicalDeviceMemoryProperties& mem_props, uint32_t type_bits,
                     VkMemoryPropertyFlags required_props, uint32_t* out_index) noexcept {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        const bool type_allowed = (type_bits & (1u << i)) != 0;
        const bool has_props =
            (mem_props.memoryTypes[i].propertyFlags & required_props) == required_props;
        if (type_allowed && has_props) {
            *out_index = i;
            return true;
        }
    }
    return false;
}

uint64_t largest_device_local_heap(const VkPhysicalDeviceMemoryProperties& mem_props) noexcept {
    uint64_t largest = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if ((mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            if (mem_props.memoryHeaps[i].size > largest) largest = mem_props.memoryHeaps[i].size;
        }
    }
    return largest;
}

// Shared by available() and Context::create() -- both need an instance
// plus the physical-device/queue-family pick, but available() tears
// everything down immediately while create() keeps it.
bool create_instance(VkInstance* out_instance) noexcept {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "boltllm";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "bolt";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    // 1.0 (not 1.4, the SDK's own version): maximizes driver compatibility
    // for this scaffolding layer -- BLLM-47's real kernels can raise this
    // later if a specific feature genuinely needs a newer API version.
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    // No extensions/validation layers: this is compute-only, off-screen
    // (no VK_KHR_surface/swapchain), and validation layers are a developer
    // opt-in, not something this scaffolding should force on unconditionally.

    return vkCreateInstance(&create_info, nullptr, out_instance) == VK_SUCCESS;
}

}  // namespace

bool available() noexcept {
    VkInstance instance = VK_NULL_HANDLE;
    if (!create_instance(&instance)) return false;

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    bool ok = false;
    if (device_count > 0) {
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
        VkPhysicalDevice picked;
        uint32_t qf;
        ok = pick_physical_device(devices, &picked, &qf);
    }

    vkDestroyInstance(instance, nullptr);
    return ok;
}

Context::~Context() noexcept { destroy(); }

bool Context::create() noexcept {
    if (device_ != nullptr) return false;  // already created -- see header's "at most once" contract

    VkInstance instance = VK_NULL_HANDLE;
    if (!create_instance(&instance)) return false;

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        vkDestroyInstance(instance, nullptr);
        return false;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    VkPhysicalDevice phys = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    if (!pick_physical_device(devices, &phys, &queue_family)) {
        vkDestroyInstance(instance, nullptr);
        return false;
    }

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    // No device extensions/features requested: this scaffolding only needs
    // buffer allocation + a compute queue, both core Vulkan 1.0.

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return false;
    }

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);

    instance_ = instance;
    physical_device_ = phys;
    device_ = device;
    queue_ = queue;
    queue_family_index_ = queue_family;
    std::memset(device_name_, 0, sizeof(device_name_));
    std::strncpy(device_name_, props.deviceName, sizeof(device_name_) - 1);
    device_local_heap_bytes_ = largest_device_local_heap(mem_props);
    return true;
}

void Context::destroy() noexcept {
    if (device_ != nullptr) {
        vkDestroyDevice(static_cast<VkDevice>(device_), nullptr);
        device_ = nullptr;
    }
    if (instance_ != nullptr) {
        vkDestroyInstance(static_cast<VkInstance>(instance_), nullptr);
        instance_ = nullptr;
    }
    physical_device_ = nullptr;
    queue_ = nullptr;
    queue_family_index_ = 0;
    std::memset(device_name_, 0, sizeof(device_name_));
    device_local_heap_bytes_ = 0;
}

bool Context::allocate_buffer(uint64_t size_bytes, bool prefer_host_visible,
                              BufferHandle* out) noexcept {
    *out = BufferHandle{};
    if (device_ == nullptr || size_bytes == 0) return false;
    VkDevice device = static_cast<VkDevice>(device_);
    VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(physical_device_);

    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size_bytes;
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &buf_info, nullptr, &buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);

    uint32_t mem_type = 0;
    bool host_visible = false;
    if (prefer_host_visible &&
        find_memory_type(mem_props, mem_reqs.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &mem_type)) {
        host_visible = true;
    } else if (find_memory_type(mem_props, mem_reqs.memoryTypeBits,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mem_type)) {
        host_visible = false;
    } else if (find_memory_type(mem_props, mem_reqs.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &mem_type)) {
        // Fallback: no pure DEVICE_LOCAL type matched (common on an
        // integrated GPU where every type is host-visible) -- host-visible
        // memory works fine as a compute buffer too, it's just also
        // mappable.
        host_visible = true;
    } else {
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    out->buffer = buffer;
    out->memory = memory;
    out->size_bytes = size_bytes;
    out->host_visible = host_visible;
    return true;
}

void Context::free_buffer(BufferHandle* handle) noexcept {
    if (handle == nullptr || device_ == nullptr) return;
    VkDevice device = static_cast<VkDevice>(device_);
    if (handle->buffer != nullptr) {
        vkDestroyBuffer(device, static_cast<VkBuffer>(handle->buffer), nullptr);
    }
    if (handle->memory != nullptr) {
        vkFreeMemory(device, static_cast<VkDeviceMemory>(handle->memory), nullptr);
    }
    *handle = BufferHandle{};
}

void* Context::map_buffer(const BufferHandle& handle) noexcept {
    if (device_ == nullptr || !handle.host_visible || handle.memory == nullptr) return nullptr;
    void* ptr = nullptr;
    const VkResult result =
        vkMapMemory(static_cast<VkDevice>(device_), static_cast<VkDeviceMemory>(handle.memory), 0,
                    handle.size_bytes, 0, &ptr);
    return result == VK_SUCCESS ? ptr : nullptr;
}

void Context::unmap_buffer(const BufferHandle& handle) noexcept {
    if (device_ == nullptr || handle.memory == nullptr) return;
    vkUnmapMemory(static_cast<VkDevice>(device_), static_cast<VkDeviceMemory>(handle.memory));
}

}  // namespace bolt::vulkan
