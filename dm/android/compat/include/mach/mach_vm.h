// Android stand-in for <mach/mach_vm.h>: the 64-bit forms of the <mach/mach.h> memory calls (dm/android/compat/include/mach/mach.h).
#pragma once
#include <mach/mach.h>
kern_return_t mach_vm_remap(vm_map_t target, mach_vm_address_t *target_addr, mach_vm_size_t size, mach_vm_offset_t mask, int flags, vm_map_t src_task,
                            mach_vm_address_t src_addr, boolean_t copy, vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inherit);
kern_return_t mach_vm_protect(vm_map_t task, mach_vm_address_t addr, mach_vm_size_t size, boolean_t set_max, vm_prot_t prot);
kern_return_t mach_vm_deallocate(vm_map_t task, mach_vm_address_t addr, mach_vm_size_t size);
kern_return_t mach_vm_region(vm_map_t task, mach_vm_address_t *addr, mach_vm_size_t *size, vm_region_flavor_t flavor, vm_region_info_t info,
                             mach_msg_type_number_t *count, mach_port_t *object);
