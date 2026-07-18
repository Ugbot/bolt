// bolt_vulkan.cpp -- implementation of bolt::vulkan's device/queue/memory
// scaffolding (BLLM-46). See bolt_vulkan.h for the full design rationale.
//
// Tiger Style adherence: no exceptions (Vulkan's C API doesn't throw
// anyway), noexcept everywhere, every failure path returns false/nullptr
// rather than asserting -- absence of Vulkan (or of a compute-capable
// device) is an expected, not exceptional, runtime condition this module
// must handle cleanly for every caller to fall back to CPU.

#include "bolt/bolt_vulkan.h"

#include <cassert>
#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

#include "bolt/vulkan_shaders/matmul_dequant_f32.h"
#include "bolt/vulkan_shaders/matmul_dequant_int2.h"
#include "bolt/vulkan_shaders/matmul_dequant_int4.h"
#include "bolt/vulkan_shaders/matmul_dequant_int8.h"
#include "bolt/vulkan_shaders/matmul_dequant_int8act_int4.h"
#include "bolt/vulkan_shaders/matmul_dequant_int8act_int4_batched.h"
#include "bolt/vulkan_shaders/matmul_dequant_int8act_int8.h"
#include "bolt/vulkan_shaders/matmul_dequant_int8act_int8_batched.h"
#include "bolt/vulkan_shaders/rmsnorm.h"  // BLLM-176 on-device RMSNorm

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
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &mem_type)) {
        // BLLM-83 UMA fast path: on an integrated/UMA GPU, one memory type
        // is typically BOTH host-visible AND device-local at once (a real,
        // measured fact on this box -- 4 of 16 memory types report all
        // three flags, backed by a genuinely separate 68GB DEVICE_LOCAL
        // heap distinct from a smaller 34GB host-only heap; verified via a
        // direct vkGetPhysicalDeviceMemoryProperties dump, not assumed).
        // Prefer this type FIRST -- find_memory_type returns the first
        // matching index, and a plain HOST_VISIBLE|HOST_COHERENT query
        // (the branch below) would have silently landed on the slower,
        // non-device-local heap purely because it happens to sort first
        // by type index, with no error or fallback ever triggering to
        // reveal that. Being also mappable this way means the "staging
        // buffer" ds4_metal.m's Shared-storage pattern exists specifically
        // to avoid was never needed here -- GpuContext.cpp already writes
        // straight into this buffer via map_buffer(), no separate
        // device-local buffer + copy step exists to eliminate.
        host_visible = true;
    } else if (prefer_host_visible &&
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
    out->device_local =
        (mem_props.memoryTypes[mem_type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
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

// ---------------------------------------------------------------------------
// BLLM-47: matmul+dequant compute pipeline.
// ---------------------------------------------------------------------------

namespace {

// F32's shader has no scale buffer (3 storage-buffer bindings: weight,
// activation, out); Int8/Int4/Int2 all have 4 (weight, scale, activation,
// out) -- see each .comp file. Centralizes that fact so create/destroy/
// dispatch never disagree on binding count.
uint32_t binding_count_for(WeightDType dtype) noexcept {
    return (dtype == WeightDType::F32) ? 3u : 4u;
}

bool shader_words_for(WeightDType dtype, const uint32_t** out_words, size_t* out_size) noexcept {
    switch (dtype) {
        case WeightDType::F32:
            *out_words = shaders::kmatmul_dequant_f32_spv;
            *out_size = shaders::kmatmul_dequant_f32_spv_size;
            return true;
        case WeightDType::Int8:
            *out_words = shaders::kmatmul_dequant_int8_spv;
            *out_size = shaders::kmatmul_dequant_int8_spv_size;
            return true;
        case WeightDType::Int4:
            *out_words = shaders::kmatmul_dequant_int4_spv;
            *out_size = shaders::kmatmul_dequant_int4_spv_size;
            return true;
        case WeightDType::Int2:
            *out_words = shaders::kmatmul_dequant_int2_spv;
            *out_size = shaders::kmatmul_dequant_int2_spv_size;
            return true;
    }
    return false;
}

// BLLM-81: shader words for the int8-ACTIVATION matmul variants (Int8/Int4
// weight only -- see bolt_vulkan.h's create_int8_activation_pipeline doc).
bool int8act_shader_words_for(WeightDType dtype, const uint32_t** out_words,
                               size_t* out_size) noexcept {
    switch (dtype) {
        case WeightDType::Int8:
            *out_words = shaders::kmatmul_dequant_int8act_int8_spv;
            *out_size = shaders::kmatmul_dequant_int8act_int8_spv_size;
            return true;
        case WeightDType::Int4:
            *out_words = shaders::kmatmul_dequant_int8act_int4_spv;
            *out_size = shaders::kmatmul_dequant_int8act_int4_spv_size;
            return true;
        case WeightDType::F32:
        case WeightDType::Int2:
            return false;  // no int8-activation hot path for these dtypes
    }
    return false;
}

// BLLM-87: shader words for the BATCHED int8-activation variants -- same
// dtype scope as int8act_shader_words_for() (Int8/Int4 only).
bool int8act_batched_shader_words_for(WeightDType dtype, const uint32_t** out_words,
                                       size_t* out_size) noexcept {
    switch (dtype) {
        case WeightDType::Int8:
            *out_words = shaders::kmatmul_dequant_int8act_int8_batched_spv;
            *out_size = shaders::kmatmul_dequant_int8act_int8_batched_spv_size;
            return true;
        case WeightDType::Int4:
            *out_words = shaders::kmatmul_dequant_int8act_int4_batched_spv;
            *out_size = shaders::kmatmul_dequant_int8act_int4_batched_spv_size;
            return true;
        case WeightDType::F32:
        case WeightDType::Int2:
            return false;
    }
    return false;
}

// Shared by create_matmul_pipeline() and create_int8_activation_pipeline():
// every field of MatmulPipeline construction is identical except which
// SPIR-V words are loaded, how many storage-buffer bindings the descriptor
// set layout needs, and the push-constant range's byte size (4 bytes for
// `{int cols}` vs 8 for `{int cols; float act_scale}`). Factored out to
// avoid ~150 lines of copy-pasted Vulkan boilerplate between the two.
bool create_pipeline_common(VkDevice device, uint32_t queue_family_index,
                             const uint32_t* spirv_words, size_t spirv_size,
                             uint32_t binding_count, uint32_t push_constant_size,
                             WeightDType dtype, MatmulPipeline* out) noexcept {
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv_size;
    module_info.pCode = spirv_words;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &module_info, nullptr, &shader_module) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[5]{};
    assert(binding_count <= 5);
    for (uint32_t i = 0; i < binding_count; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = binding_count;
    set_layout_info.pBindings = bindings;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_constant_size;

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) !=
        VK_SUCCESS) {
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = binding_count;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    VkDescriptorSetAllocateInfo set_alloc_info{};
    set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc_info.descriptorPool = descriptor_pool;
    set_alloc_info.descriptorSetCount = 1;
    set_alloc_info.pSetLayouts = &set_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &set_alloc_info, &descriptor_set) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    VkCommandPoolCreateInfo cmd_pool_info{};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmd_pool_info.queueFamilyIndex = queue_family_index;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &cmd_pool_info, nullptr, &command_pool) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    VkCommandBufferAllocateInfo cmd_alloc_info{};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = command_pool;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &cmd_alloc_info, &command_buffer) != VK_SUCCESS) {
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        return false;
    }

    out->shader_module = shader_module;
    out->descriptor_set_layout = set_layout;
    out->pipeline_layout = pipeline_layout;
    out->pipeline = pipeline;
    out->descriptor_pool = descriptor_pool;
    out->descriptor_set = descriptor_set;
    out->command_pool = command_pool;
    out->command_buffer = command_buffer;
    out->dtype = dtype;
    return true;
}

}  // namespace

bool Context::create_matmul_pipeline(WeightDType dtype, MatmulPipeline* out) noexcept {
    *out = MatmulPipeline{};
    if (device_ == nullptr) return false;

    const uint32_t* spirv_words = nullptr;
    size_t spirv_size = 0;
    if (!shader_words_for(dtype, &spirv_words, &spirv_size)) return false;

    return create_pipeline_common(static_cast<VkDevice>(device_), queue_family_index_, spirv_words,
                                   spirv_size, binding_count_for(dtype), sizeof(int32_t), dtype, out);
}

bool Context::create_int8_activation_pipeline(WeightDType dtype, MatmulPipeline* out) noexcept {
    *out = MatmulPipeline{};
    if (device_ == nullptr) return false;

    const uint32_t* spirv_words = nullptr;
    size_t spirv_size = 0;
    if (!int8act_shader_words_for(dtype, &spirv_words, &spirv_size)) return false;

    // Both int8-activation shaders always have 4 bindings (weight,
    // weight_scale, activation, out) and an 8-byte push constant
    // (`{int cols; float act_scale;}`).
    return create_pipeline_common(static_cast<VkDevice>(device_), queue_family_index_, spirv_words,
                                   spirv_size, /*binding_count=*/4, /*push_constant_size=*/8, dtype,
                                   out);
}

bool Context::create_int8_activation_batched_pipeline(WeightDType dtype,
                                                        MatmulPipeline* out) noexcept {
    *out = MatmulPipeline{};
    if (device_ == nullptr) return false;

    const uint32_t* spirv_words = nullptr;
    size_t spirv_size = 0;
    if (!int8act_batched_shader_words_for(dtype, &spirv_words, &spirv_size)) return false;

    // Both batched int8-activation shaders always have 5 bindings (weight,
    // weight_scale, activation, act_scale, out) and an 8-byte push constant
    // (`{int cols; int rows;}`).
    return create_pipeline_common(static_cast<VkDevice>(device_), queue_family_index_, spirv_words,
                                   spirv_size, /*binding_count=*/5, /*push_constant_size=*/8, dtype,
                                   out);
}

bool Context::create_rmsnorm_pipeline(MatmulPipeline* out) noexcept {
    *out = MatmulPipeline{};
    if (device_ == nullptr) return false;
    // BLLM-176: 3 bindings (x, weight, out) + 8-byte push constant {int n; float eps;}.
    return create_pipeline_common(static_cast<VkDevice>(device_), queue_family_index_,
                                   shaders::krmsnorm_spv, shaders::krmsnorm_spv_size,
                                   /*binding_count=*/3, /*push_constant_size=*/8, WeightDType::F32,
                                   out);
}

bool Context::rmsnorm(const MatmulPipeline& pipeline, const BufferHandle& x,
                      const BufferHandle& weight, const BufferHandle& out, int32_t n,
                      float eps) noexcept {
    if (device_ == nullptr || !pipeline.is_valid()) return false;
    if (x.buffer == nullptr || weight.buffer == nullptr || out.buffer == nullptr || n <= 0) {
        return false;
    }
    VkDevice device = static_cast<VkDevice>(device_);

    VkDescriptorBufferInfo buffer_infos[3] = {
        {static_cast<VkBuffer>(x.buffer), 0, VK_WHOLE_SIZE},
        {static_cast<VkBuffer>(weight.buffer), 0, VK_WHOLE_SIZE},
        {static_cast<VkBuffer>(out.buffer), 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = static_cast<VkDescriptorSet>(pipeline.descriptor_set);
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(pipeline.command_buffer);
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) return false;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      static_cast<VkPipeline>(pipeline.pipeline));
    VkDescriptorSet set = static_cast<VkDescriptorSet>(pipeline.descriptor_set);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            static_cast<VkPipelineLayout>(pipeline.pipeline_layout), 0, 1, &set, 0,
                            nullptr);
    struct {
        int32_t n;
        float eps;
    } pc{n, eps};
    vkCmdPushConstants(cmd, static_cast<VkPipelineLayout>(pipeline.pipeline_layout),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);  // one workgroup normalizes the whole vector
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    VkQueue queue = static_cast<VkQueue>(queue_);
    if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) return false;
    return vkQueueWaitIdle(queue) == VK_SUCCESS;
}

void Context::destroy_matmul_pipeline(MatmulPipeline* pipeline) noexcept {
    if (pipeline == nullptr || device_ == nullptr) return;
    VkDevice device = static_cast<VkDevice>(device_);

    // Command buffers are freed implicitly when their pool is destroyed;
    // descriptor sets likewise when their pool is destroyed -- only the
    // pools themselves need an explicit destroy call.
    if (pipeline->command_pool != nullptr) {
        vkDestroyCommandPool(device, static_cast<VkCommandPool>(pipeline->command_pool), nullptr);
    }
    if (pipeline->descriptor_pool != nullptr) {
        vkDestroyDescriptorPool(device, static_cast<VkDescriptorPool>(pipeline->descriptor_pool),
                                 nullptr);
    }
    if (pipeline->pipeline != nullptr) {
        vkDestroyPipeline(device, static_cast<VkPipeline>(pipeline->pipeline), nullptr);
    }
    if (pipeline->pipeline_layout != nullptr) {
        vkDestroyPipelineLayout(device, static_cast<VkPipelineLayout>(pipeline->pipeline_layout),
                                 nullptr);
    }
    if (pipeline->descriptor_set_layout != nullptr) {
        vkDestroyDescriptorSetLayout(
            device, static_cast<VkDescriptorSetLayout>(pipeline->descriptor_set_layout), nullptr);
    }
    if (pipeline->shader_module != nullptr) {
        vkDestroyShaderModule(device, static_cast<VkShaderModule>(pipeline->shader_module), nullptr);
    }
    *pipeline = MatmulPipeline{};
}

void Context::destroy_int8_activation_pipeline(MatmulPipeline* pipeline) noexcept {
    // Identical teardown to destroy_matmul_pipeline() -- MatmulPipeline's
    // resources are torn down the same way regardless of which shader
    // family created them.
    destroy_matmul_pipeline(pipeline);
}

void Context::destroy_int8_activation_batched_pipeline(MatmulPipeline* pipeline) noexcept {
    destroy_matmul_pipeline(pipeline);
}

bool Context::matmul_dequant(const MatmulPipeline& pipeline, const BufferHandle& weight,
                             const BufferHandle& scale, const BufferHandle& activation,
                             const BufferHandle& out, int32_t out_rows, int32_t cols) noexcept {
    if (device_ == nullptr || !pipeline.is_valid()) return false;
    if (weight.buffer == nullptr || activation.buffer == nullptr || out.buffer == nullptr) {
        return false;
    }
    if (pipeline.dtype != WeightDType::F32 && scale.buffer == nullptr) return false;
    if (out_rows <= 0 || cols <= 0) return false;

    VkDevice device = static_cast<VkDevice>(device_);
    const uint32_t binding_count = binding_count_for(pipeline.dtype);

    VkDescriptorBufferInfo buffer_infos[4]{};
    buffer_infos[0] = {static_cast<VkBuffer>(weight.buffer), 0, VK_WHOLE_SIZE};
    if (pipeline.dtype == WeightDType::F32) {
        buffer_infos[1] = {static_cast<VkBuffer>(activation.buffer), 0, VK_WHOLE_SIZE};
        buffer_infos[2] = {static_cast<VkBuffer>(out.buffer), 0, VK_WHOLE_SIZE};
    } else {
        buffer_infos[1] = {static_cast<VkBuffer>(scale.buffer), 0, VK_WHOLE_SIZE};
        buffer_infos[2] = {static_cast<VkBuffer>(activation.buffer), 0, VK_WHOLE_SIZE};
        buffer_infos[3] = {static_cast<VkBuffer>(out.buffer), 0, VK_WHOLE_SIZE};
    }

    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < binding_count; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = static_cast<VkDescriptorSet>(pipeline.descriptor_set);
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device, binding_count, writes, 0, nullptr);

    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(pipeline.command_buffer);
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) return false;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<VkPipeline>(pipeline.pipeline));
    VkDescriptorSet set = static_cast<VkDescriptorSet>(pipeline.descriptor_set);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            static_cast<VkPipelineLayout>(pipeline.pipeline_layout), 0, 1, &set, 0,
                            nullptr);
    vkCmdPushConstants(cmd, static_cast<VkPipelineLayout>(pipeline.pipeline_layout),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t), &cols);
    vkCmdDispatch(cmd, static_cast<uint32_t>(out_rows), 1, 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    VkQueue queue = static_cast<VkQueue>(queue_);
    if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) return false;

    // Blocking: simplest correct synchronization for a first real cut (no
    // async dispatch/fence-polling API yet -- same "block until done"
    // contract as bolt::rocm::matmul_dequant's hipDeviceSynchronize()).
    // vkQueueWaitIdle establishes the host-visibility guarantee needed to
    // safely map_buffer()+read `out` afterward on this HOST_COHERENT
    // memory, so no separate pipeline barrier is required.
    return vkQueueWaitIdle(queue) == VK_SUCCESS;
}

bool Context::matmul_dequant_int8_activation(const MatmulPipeline& pipeline,
                                             const BufferHandle& weight,
                                             const BufferHandle& weight_scale,
                                             const BufferHandle& activation, float act_scale,
                                             const BufferHandle& out, int32_t out_rows,
                                             int32_t cols) noexcept {
    if (device_ == nullptr || !pipeline.is_valid()) return false;
    if (pipeline.dtype != WeightDType::Int8 && pipeline.dtype != WeightDType::Int4) return false;
    if (weight.buffer == nullptr || weight_scale.buffer == nullptr || activation.buffer == nullptr ||
        out.buffer == nullptr) {
        return false;
    }
    if (out_rows <= 0 || cols <= 0) return false;

    VkDevice device = static_cast<VkDevice>(device_);

    VkDescriptorBufferInfo buffer_infos[4]{};
    buffer_infos[0] = {static_cast<VkBuffer>(weight.buffer), 0, VK_WHOLE_SIZE};
    buffer_infos[1] = {static_cast<VkBuffer>(weight_scale.buffer), 0, VK_WHOLE_SIZE};
    buffer_infos[2] = {static_cast<VkBuffer>(activation.buffer), 0, VK_WHOLE_SIZE};
    buffer_infos[3] = {static_cast<VkBuffer>(out.buffer), 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = static_cast<VkDescriptorSet>(pipeline.descriptor_set);
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(pipeline.command_buffer);
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) return false;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<VkPipeline>(pipeline.pipeline));
    VkDescriptorSet set = static_cast<VkDescriptorSet>(pipeline.descriptor_set);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            static_cast<VkPipelineLayout>(pipeline.pipeline_layout), 0, 1, &set, 0,
                            nullptr);
    // Push constants layout matches each int8act shader's
    // `PushConstants { int cols; float act_scale; }` exactly (offset 0 =
    // cols, offset 4 = act_scale) -- packed into one struct so a single
    // vkCmdPushConstants call covers both fields.
    struct PushConstants {
        int32_t cols;
        float act_scale;
    };
    const PushConstants push{cols, act_scale};
    vkCmdPushConstants(cmd, static_cast<VkPipelineLayout>(pipeline.pipeline_layout),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);
    vkCmdDispatch(cmd, static_cast<uint32_t>(out_rows), 1, 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    VkQueue queue = static_cast<VkQueue>(queue_);
    if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) return false;

    return vkQueueWaitIdle(queue) == VK_SUCCESS;
}

bool Context::matmul_dequant_int8_activation_batched(const MatmulPipeline& pipeline,
                                                       const BufferHandle& weight,
                                                       const BufferHandle& weight_scale,
                                                       const BufferHandle& activation,
                                                       const BufferHandle& act_scale,
                                                       const BufferHandle& out, int32_t out_rows,
                                                       int32_t cols, int32_t n_tokens) noexcept {
    if (device_ == nullptr || !pipeline.is_valid()) return false;
    if (pipeline.dtype != WeightDType::Int8 && pipeline.dtype != WeightDType::Int4) return false;
    if (weight.buffer == nullptr || weight_scale.buffer == nullptr || activation.buffer == nullptr ||
        act_scale.buffer == nullptr || out.buffer == nullptr) {
        return false;
    }
    if (out_rows <= 0 || cols <= 0 || n_tokens <= 0) return false;

    VkDevice device = static_cast<VkDevice>(device_);

    VkDescriptorBufferInfo buffer_infos[5]{};
    buffer_infos[0] = {static_cast<VkBuffer>(weight.buffer), 0, VK_WHOLE_SIZE};
    buffer_infos[1] = {static_cast<VkBuffer>(weight_scale.buffer), 0, VK_WHOLE_SIZE};
    buffer_infos[2] = {static_cast<VkBuffer>(activation.buffer), 0, VK_WHOLE_SIZE};
    buffer_infos[3] = {static_cast<VkBuffer>(act_scale.buffer), 0, VK_WHOLE_SIZE};
    buffer_infos[4] = {static_cast<VkBuffer>(out.buffer), 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = static_cast<VkDescriptorSet>(pipeline.descriptor_set);
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);

    VkCommandBuffer cmd = static_cast<VkCommandBuffer>(pipeline.command_buffer);
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) return false;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<VkPipeline>(pipeline.pipeline));
    VkDescriptorSet set = static_cast<VkDescriptorSet>(pipeline.descriptor_set);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            static_cast<VkPipelineLayout>(pipeline.pipeline_layout), 0, 1, &set, 0,
                            nullptr);
    // Push constants layout matches the batched shaders' own
    // `PushConstants { int cols; int rows; }` exactly.
    struct PushConstants {
        int32_t cols;
        int32_t rows;
    };
    const PushConstants push{cols, out_rows};
    vkCmdPushConstants(cmd, static_cast<VkPipelineLayout>(pipeline.pipeline_layout),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push);
    // Dispatch shape (out_rows, n_tokens, 1) -- gl_WorkGroupID.x selects the
    // output row, gl_WorkGroupID.y selects the token (see the .comp file's
    // own header for this convention).
    vkCmdDispatch(cmd, static_cast<uint32_t>(out_rows), static_cast<uint32_t>(n_tokens), 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    VkQueue queue = static_cast<VkQueue>(queue_);
    if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) return false;

    return vkQueueWaitIdle(queue) == VK_SUCCESS;
}

}  // namespace bolt::vulkan
