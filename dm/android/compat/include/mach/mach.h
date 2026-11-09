// Android stand-in for <mach/mach.h>: the Mach calls the host makes, on Linux (implemented in dm/android/compat/dm_compat.c).
// - Memory: vm_read_overwrite / vm_write are fault-safe copies within this process (process_vm_readv / writev, which report an unmapped page as an error instead
//   of faulting); vm_region_64 reads /proc/self/maps; vm_allocate / vm_deallocate / vm_protect are mmap / munmap / mprotect.
// - mach_vm_remap / vm_remap (a second mapping of the same pages) has no Linux form for private anonymous memory: it fails, and the engine's callers (cheats'
//   writable alias of main, a probe's watch alias) already handle the failure. The iPhone loader path that needs it is not built on Android.
// - Thread inspection (thread_suspend / thread_get_state / task_threads, used only by the stall reports, backtraces and the sampler) fails: those diagnostics
//   report nothing on Android until they get a signal-based form.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <mach/mach_time.h>

typedef int kern_return_t;
typedef unsigned int natural_t;
typedef int integer_t;
typedef uintptr_t mach_port_t;
typedef mach_port_t task_t, thread_act_t, thread_t, vm_map_t, mach_port_name_t;
typedef uintptr_t vm_address_t, vm_size_t, vm_offset_t;
typedef uint64_t mach_vm_address_t, mach_vm_size_t, mach_vm_offset_t;
typedef int vm_prot_t, vm_inherit_t, vm_behavior_t;
typedef int boolean_t;
typedef natural_t mach_msg_type_number_t;
typedef thread_act_t *thread_act_array_t;
typedef natural_t *thread_state_t;
typedef integer_t *thread_info_t;
typedef int *vm_region_info_t;
typedef int vm_region_flavor_t, thread_flavor_t;

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#define KERN_SUCCESS 0
#define KERN_INVALID_ADDRESS 1
#define KERN_PROTECTION_FAILURE 2
#define KERN_NO_SPACE 3
#define KERN_INVALID_ARGUMENT 4
#define KERN_FAILURE 5
#define KERN_NOT_SUPPORTED 46
#define MACH_PORT_NULL 0
#define VM_PROT_NONE 0
#define VM_PROT_READ 1
#define VM_PROT_WRITE 2
#define VM_PROT_EXECUTE 4
#define VM_PROT_ALL 7
#define VM_FLAGS_FIXED 0
#define VM_FLAGS_ANYWHERE 1
#define VM_FLAGS_OVERWRITE 0x4000
#define VM_INHERIT_SHARE 0
#define VM_INHERIT_COPY 1
#define VM_INHERIT_NONE 2

typedef struct {
    vm_prot_t protection, max_protection;
    vm_inherit_t inheritance;
    boolean_t shared, reserved;
    uint64_t offset;
    vm_behavior_t behavior;
    unsigned short user_wired_count;
} vm_region_basic_info_data_64_t;
typedef vm_region_basic_info_data_64_t vm_region_basic_info_64_t;
#define VM_REGION_BASIC_INFO_64 9
#define VM_REGION_BASIC_INFO_COUNT_64 ((mach_msg_type_number_t)(sizeof(vm_region_basic_info_data_64_t) / sizeof(int)))

typedef struct { int seconds, microseconds; } time_value_t;
typedef struct {
    time_value_t user_time, system_time;
    integer_t cpu_usage, policy, run_state, flags, suspend_count, sleep_time;
} thread_basic_info_data_t;
typedef thread_basic_info_data_t *thread_basic_info_t;
#define THREAD_BASIC_INFO 3
#define THREAD_BASIC_INFO_COUNT ((mach_msg_type_number_t)(sizeof(thread_basic_info_data_t) / sizeof(natural_t)))
#define TH_STATE_RUNNING 1
#define TH_STATE_STOPPED 2
#define TH_STATE_WAITING 3
#define TH_STATE_UNINTERRUPTIBLE 4
#define TH_STATE_HALTED 5

// Apple's arm64 thread state and its accessors (the stall reports read it after thread_get_state, which always fails here)
typedef struct { uint64_t __x[29], __fp, __lr, __sp, __pc; uint32_t __cpsr, __pad; } arm_thread_state64_t;
#define __darwin_arm_thread_state64_get_pc(s) ((s).__pc)
#define __darwin_arm_thread_state64_get_lr(s) ((s).__lr)
#define __darwin_arm_thread_state64_get_sp(s) ((s).__sp)
#define __darwin_arm_thread_state64_get_fp(s) ((s).__fp)
#define ARM_THREAD_STATE64 6
#define ARM_THREAD_STATE64_COUNT ((mach_msg_type_number_t)(sizeof(arm_thread_state64_t) / sizeof(natural_t)))

#include <unistd.h>
#define vm_page_size ((vm_size_t)getpagesize())
#define vm_page_mask (vm_page_size - 1)
static inline mach_port_t mach_task_self(void) { return 1; }
static inline mach_port_t mach_thread_self(void) { return 0; }
static inline kern_return_t mach_port_deallocate(task_t t, mach_port_name_t n) { (void)t; (void)n; return KERN_SUCCESS; }

kern_return_t vm_read_overwrite(vm_map_t task, vm_address_t addr, vm_size_t size, vm_address_t data, vm_size_t *out_size);
kern_return_t vm_write(vm_map_t task, vm_address_t addr, vm_offset_t data, mach_msg_type_number_t count);
kern_return_t vm_allocate(vm_map_t task, vm_address_t *addr, vm_size_t size, int flags);
kern_return_t vm_deallocate(vm_map_t task, vm_address_t addr, vm_size_t size);
kern_return_t vm_protect(vm_map_t task, vm_address_t addr, vm_size_t size, boolean_t set_max, vm_prot_t prot);
kern_return_t vm_region_64(vm_map_t task, vm_address_t *addr, vm_size_t *size, vm_region_flavor_t flavor, vm_region_info_t info, mach_msg_type_number_t *count, mach_port_t *object);
kern_return_t vm_remap(vm_map_t target, vm_address_t *target_addr, vm_size_t size, vm_address_t mask, int flags, vm_map_t src_task, vm_address_t src_addr,
                       boolean_t copy, vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inherit);

kern_return_t thread_suspend(thread_act_t t);
kern_return_t thread_resume(thread_act_t t);
kern_return_t thread_get_state(thread_act_t t, thread_flavor_t flavor, thread_state_t state, mach_msg_type_number_t *count);
kern_return_t thread_info(thread_act_t t, thread_flavor_t flavor, thread_info_t info, mach_msg_type_number_t *count);
kern_return_t task_threads(task_t task, thread_act_array_t *list, mach_msg_type_number_t *count);

// task_info: TASK_VM_INFO answers the resident memory (phys_footprint and resident_size from /proc/self/statm: the host's memory readout and its memory
// watchdog read it); TASK_POWER_INFO_V2 (energy counters) has no Linux source and fails, so the power lines are not printed
typedef natural_t task_flavor_t;
typedef integer_t *task_info_t;
typedef struct {
    uint64_t virtual_size; integer_t region_count, page_size;
    uint64_t resident_size, resident_size_peak, device, device_peak, internal, internal_peak, external, external_peak, reusable, reusable_peak,
             purgeable_volatile_pmap, purgeable_volatile_resident, purgeable_volatile_virtual, compressed, compressed_peak, compressed_lifetime, phys_footprint;
} task_vm_info_data_t;
#define TASK_VM_INFO 22
#define TASK_VM_INFO_COUNT ((mach_msg_type_number_t)(sizeof(task_vm_info_data_t) / sizeof(natural_t)))
typedef struct { uint64_t total_user, total_system, task_interrupt_wakeups, task_platform_idle_wakeups, task_timer_wakeups_bin_1, task_timer_wakeups_bin_2; } task_power_info_data_t;
typedef struct { uint64_t task_gpu_utilisation, task_gpu_stat_reserved0, task_gpu_stat_reserved1, task_gpu_stat_reserved2; } gpu_energy_data;
typedef struct { task_power_info_data_t cpu_energy; gpu_energy_data gpu_energy; uint64_t task_energy, task_ptime, task_pset_switches; } task_power_info_v2_data_t;
#define TASK_POWER_INFO_V2 26
#define TASK_POWER_INFO_V2_COUNT ((mach_msg_type_number_t)(sizeof(task_power_info_v2_data_t) / sizeof(natural_t)))
kern_return_t task_info(task_t task, task_flavor_t flavor, task_info_t info, mach_msg_type_number_t *count);

#include <pthread.h>
static inline mach_port_t pthread_mach_thread_np(pthread_t t) { (void)t; return 0; }   // 0: the callers skip thread inspection for a thread without a port
#include <mach/mach_vm.h>
