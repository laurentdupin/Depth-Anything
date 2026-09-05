#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct dav1_context;
ibr_linux_capture_capabilities dav1_linux_capture_capabilities(dav1_context *);
void dav1_infer_linux_capture(
    dav1_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint32_t, float *);
#endif
