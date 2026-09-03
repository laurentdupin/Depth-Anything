#include "depth_anything_v1.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static double now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
#endif
}

static int parse_u32(const char* text, uint32_t* value, int allow_zero) {
    char* end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (text == end || *end != '\0' || (!allow_zero && parsed == 0) ||
        parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int parse_encoder(const char* text, dav1_encoder* encoder) {
    if (strcmp(text, "vits") == 0) *encoder = DAV1_ENCODER_VITS;
    else if (strcmp(text, "vitb") == 0) *encoder = DAV1_ENCODER_VITB;
    else if (strcmp(text, "vitl") == 0) *encoder = DAV1_ENCODER_VITL;
    else return 0;
    return 1;
}

static int parse_precision(const char* text, uint32_t* flags) {
    if (strcmp(text, "fp32") == 0) *flags = 0;
    else if (strcmp(text, "fp16") == 0) *flags = DAV1_CREATE_FORCE_FP16;
    else if (strcmp(text, "int8") == 0) *flags = DAV1_CREATE_FORCE_INT8;
    else return 0;
    *flags |= DAV1_CREATE_FORCE_VULKAN;
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s MODEL ENCODER WIDTH HEIGHT PRECISION "
            "WARMUP ITERATIONS DEVICE\n", argv[0]);
        return 2;
    }
    dav1_encoder encoder;
    uint32_t width, height, flags, warmup, iterations, device;
    if (!parse_encoder(argv[2], &encoder) ||
        !parse_u32(argv[3], &width, 0) ||
        !parse_u32(argv[4], &height, 0) ||
        !parse_precision(argv[5], &flags) ||
        !parse_u32(argv[6], &warmup, 1) ||
        !parse_u32(argv[7], &iterations, 0) ||
        !parse_u32(argv[8], &device, 1) || width % 14 || height % 14) {
        fprintf(stderr, "invalid arguments; dimensions must be multiples of 14\n");
        return 2;
    }
    const size_t pixels = (size_t)width * height;
    if (pixels > SIZE_MAX / (3u * sizeof(float))) return 2;
    float* input = (float*)malloc(pixels * 3u * sizeof(float));
    float* output = (float*)malloc(pixels * sizeof(float));
    if (!input || !output) return 3;
    for (size_t index = 0; index < pixels * 3u; ++index)
        input[index] = (float)((index * 1103515245u + 12345u) & 1023u) /
            1023.0f;
    const dav1_create_options options = {
        sizeof(options), DAV1_ABI_VERSION, encoder, (int32_t)device, flags};
    dav1_context* context = NULL;
    const double load_begin = now_ms();
    dav1_status status = dav1_create(argv[1], &options, &context);
    const double load_ms = now_ms() - load_begin;
    if (status != DAV1_STATUS_OK) {
        fprintf(stderr, "load failed: %s\n", dav1_last_error());
        return 4;
    }
    for (uint32_t index = 0; index < warmup; ++index) {
        status = dav1_infer_tensor_f32(
            context, input, (int32_t)width, (int32_t)height, output, pixels);
        if (status != DAV1_STATUS_OK) return 5;
    }
    double total = 0.0, minimum = 1.0e30, maximum = 0.0;
    for (uint32_t index = 0; index < iterations; ++index) {
        const double begin = now_ms();
        status = dav1_infer_tensor_f32(
            context, input, (int32_t)width, (int32_t)height, output, pixels);
        const double elapsed = now_ms() - begin;
        if (status != DAV1_STATUS_OK) return 6;
        total += elapsed;
        if (elapsed < minimum) minimum = elapsed;
        if (elapsed > maximum) maximum = elapsed;
    }
    float output_min = INFINITY, output_max = -INFINITY;
    double output_sum = 0.0;
    for (size_t index = 0; index < pixels; ++index) {
        if (output[index] < output_min) output_min = output[index];
        if (output[index] > output_max) output_max = output[index];
        output_sum += output[index];
    }
    const double mean = total / iterations;
    printf("load_ms=%.3f width=%u height=%u warmup=%u iterations=%u "
        "mean_ms=%.3f min_ms=%.3f max_ms=%.3f fps=%.3f "
        "output_min=%.9g output_max=%.9g output_mean=%.9g\n",
        load_ms, width, height, warmup, iterations, mean, minimum, maximum,
        1000.0 / mean, output_min, output_max, output_sum / pixels);
    dav1_destroy(context);
    free(output);
    free(input);
    return 0;
}
