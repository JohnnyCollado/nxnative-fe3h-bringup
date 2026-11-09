// Android stand-in for <mach/thread_policy.h>: the present thread's real-time ("time constraint") policy and the thread status report (hle_nvn.c). Linux has no
// such policy for an app, so setting and reading it fail (KERN_NOT_SUPPORTED): hle_nvn.c logs "failed to set" and keeps running on the standard policy.
#pragma once
#include <mach/mach.h>
typedef natural_t thread_policy_flavor_t;
typedef integer_t *thread_policy_t;
typedef struct { uint32_t period, computation, constraint; boolean_t preemptible; } thread_time_constraint_policy_data_t;
#define THREAD_TIME_CONSTRAINT_POLICY 2
#define THREAD_TIME_CONSTRAINT_POLICY_COUNT ((mach_msg_type_number_t)(sizeof(thread_time_constraint_policy_data_t) / sizeof(integer_t)))
typedef struct {
    uint64_t pth_user_time, pth_system_time;
    int32_t pth_cpu_usage, pth_policy, pth_run_state, pth_flags, pth_sleep_time, pth_curpri, pth_priority, pth_maxpriority;
    char pth_name[64];
} thread_extended_info_data_t;
#define THREAD_EXTENDED_INFO 5
#define THREAD_EXTENDED_INFO_COUNT ((mach_msg_type_number_t)(sizeof(thread_extended_info_data_t) / sizeof(natural_t)))
static inline kern_return_t thread_policy_set(thread_act_t t, thread_policy_flavor_t f, thread_policy_t p, mach_msg_type_number_t c) { (void)t; (void)f; (void)p; (void)c; return KERN_NOT_SUPPORTED; }
static inline kern_return_t thread_policy_get(thread_act_t t, thread_policy_flavor_t f, thread_policy_t p, mach_msg_type_number_t *c, boolean_t *def) { (void)t; (void)f; (void)p; (void)c; (void)def; return KERN_NOT_SUPPORTED; }
