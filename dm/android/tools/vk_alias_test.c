// Can GPU-visible Vulkan memory sit at an address the host chooses? The Android host's Vulkan backend needs the game's NVN memory pools, which live at addresses
// the game picked in its own heap, to be memory the GPU reads and writes, as the Mac backend gets with Metal buffers made over that memory. The Adreno driver has no
// VK_EXT_external_memory_host (vk_probe.c), so this tests three ways of putting driver memory at a chosen address instead (the pool's content is copied in first):
//   1 export:  memory allocated exportable as an opaque fd (VK_KHR_external_memory_fd), the fd mapped at the address (mmap MAP_SHARED | MAP_FIXED)
//   2 alias:   a second view of the driver's own mapping (vkMapMemory) at the address: mremap with old size 0 (Linux duplicates a shared mapping)
//   3 move:    the driver's mapping moved to the address (mremap MREMAP_MAYMOVE | MREMAP_FIXED)
// Each passes when what the CPU writes through the chosen address is what the GPU sees (a buffer copy reads it), and what the GPU writes (a fill) is what the CPU reads there.
// Build: NDK clang --target=aarch64-linux-android35 vk_alias_test.c -lvulkan -o vk_alias_test
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define SZ (8u << 20)
static VkDevice dev; static VkQueue q; static VkCommandPool cp; static VkPhysicalDeviceMemoryProperties mp;

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}
// GPU: copy 4 KB from offset 0 to offset 64 KB, then fill 4 KB at 128 KB with a pattern
static void gpu_round(VkBuffer b) {
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev, &ai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    VkBufferCopy c = {0, 65536, 4096}; vkCmdCopyBuffer(cb, b, b, 1, &c);
    vkCmdFillBuffer(cb, b, 131072, 4096, 0xCAFEF00Du);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
    vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(q); vkFreeCommandBuffers(dev, cp, 1, &cb);
}
// The check through the chosen address: the CPU's bytes copied by the GPU, the GPU's fill read by the CPU
static int check(uint8_t *at) {
    for (int i = 0; i < 4096; i++) at[i] = (uint8_t)(i * 7 + 3);
    return 0;
}
static int verify(uint8_t *at) {
    for (int i = 0; i < 4096; i++) if (at[65536 + i] != (uint8_t)(i * 7 + 3)) { printf("    the GPU did not see the CPU's bytes (offset %d: %#x)\n", i, at[65536 + i]); return 0; }
    for (int i = 0; i < 1024; i++) if (((uint32_t *)(at + 131072))[i] != 0xCAFEF00Du) { printf("    the CPU did not see the GPU's fill (word %d: %#x)\n", i, ((uint32_t *)(at + 131072))[i]); return 0; }
    return 1;
}
static VkBuffer make_buffer(VkDeviceMemory m, const void *ext) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, ext, 0, SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b; if (vkCreateBuffer(dev, &bci, NULL, &b)) return VK_NULL_HANDLE;
    if (vkBindBufferMemory(dev, b, m, 0)) return VK_NULL_HANDLE;
    return b;
}
// The address the game's pool would be at: reserved, with old content the alias must replace
static uint8_t *game_address(void) {
    uint8_t *a = mmap(NULL, SZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(a, 0x5a, SZ); return a;
}

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "alias", 1, "nxnative", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst; vkCreateInstance(&ici, NULL, &inst);
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &n, &pd);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float prio = 1; VkDeviceQueueCreateInfo dqi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    const char *ext[] = {"VK_KHR_external_memory_fd"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &dqi, 0, NULL, 1, ext, NULL};
    if (vkCreateDevice(pd, &dci, NULL, &dev)) { printf("vkCreateDevice failed\n"); return 1; }
    vkGetDeviceQueue(dev, 0, 0, &q);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0}; vkCreateCommandPool(dev, &cpi, NULL, &cp);
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int pass = 0;

    // 1 export
    {
        printf("1 export: opaque fd memory mapped at the chosen address\n");
        VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, 0, SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer tmp; vkCreateBuffer(dev, &bci, NULL, &tmp); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, tmp, &mr); vkDestroyBuffer(dev, tmp, NULL);
        VkExportMemoryAllocateInfo ex = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &ex, mr.size, mem_type(mr.memoryTypeBits, want)};
        VkDeviceMemory m; VkResult r = vkAllocateMemory(dev, &mai, NULL, &m);
        PFN_vkGetMemoryFdKHR gf = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
        int fd = -1; VkMemoryGetFdInfoKHR gi = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, NULL, m, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        if (!r) r = gf(dev, &gi, &fd);
        printf("  allocate + export: %d, fd %d\n", r, fd);
        if (!r) {
            uint8_t *at = game_address();
            void *p = mmap(at, SZ, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
            if (p == MAP_FAILED) printf("  mmap of the fd at the address: failed, errno %d (%s)\n", errno, strerror(errno));
            else { VkBuffer b = make_buffer(m, &eb); check(at); if (b) gpu_round(b); int ok = b && verify(at); printf("  %s\n", ok ? "PASS" : "fail"); pass |= ok; }
        }
    }
    // 2 alias and 3 move: the driver's own mapping
    for (int mode = 2; mode <= 3; mode++) {
        printf("%d %s\n", mode, mode == 2 ? "alias: a second view of vkMapMemory's mapping (mremap old size 0)" : "move: vkMapMemory's mapping moved (mremap MREMAP_FIXED)");
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer tmp; vkCreateBuffer(dev, &bci, NULL, &tmp); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, tmp, &mr); vkDestroyBuffer(dev, tmp, NULL);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mem_type(mr.memoryTypeBits, want)};
        VkDeviceMemory m; void *drv = NULL;
        if (vkAllocateMemory(dev, &mai, NULL, &m) || vkMapMemory(dev, m, 0, VK_WHOLE_SIZE, 0, &drv)) { printf("  allocate / map failed\n"); continue; }
        uint8_t *at = game_address();
        void *p = mremap(drv, mode == 2 ? 0 : SZ, SZ, MREMAP_MAYMOVE | MREMAP_FIXED, at);
        if (p == MAP_FAILED) { printf("  mremap failed: errno %d (%s)\n", errno, strerror(errno)); continue; }
        VkBuffer b = make_buffer(m, NULL); check(at); if (b) gpu_round(b);
        int ok = b && verify(at);
        if (mode == 2 && ok) ok = ((uint8_t *)drv)[65536 + 5] == at[65536 + 5];   // the driver's view and the alias are the same pages
        printf("  %s\n", ok ? "PASS" : "fail"); pass |= ok;
    }
    printf("%s\n", pass ? "at least one way works" : "no way works: the backend must copy pool memory");
    return 0;
}
