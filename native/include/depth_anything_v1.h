#ifndef DEPTH_ANYTHING_V1_H
#define DEPTH_ANYTHING_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(DAV1_BUILD_DLL)
#    define DAV1_API __declspec(dllexport)
#  else
#    define DAV1_API __declspec(dllimport)
#  endif
#  define DAV1_CALL __cdecl
#else
#  define DAV1_API __attribute__((visibility("default")))
#  define DAV1_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DAV1_ABI_VERSION 1u

typedef struct dav1_context dav1_context;
typedef struct dav1_gpu_job dav1_gpu_job;
typedef struct dav1_gpu_output_lease dav1_gpu_output_lease;

typedef enum dav1_status {
    DAV1_STATUS_OK = 0,
    DAV1_STATUS_INVALID_ARGUMENT = 1,
    DAV1_STATUS_MODEL_IO = 2,
    DAV1_STATUS_MODEL_FORMAT = 3,
    DAV1_STATUS_VULKAN_UNAVAILABLE = 4,
    DAV1_STATUS_OUT_OF_MEMORY = 5,
    DAV1_STATUS_INFERENCE_FAILED = 6,
    DAV1_STATUS_BUFFER_TOO_SMALL = 7,
    DAV1_STATUS_UNSUPPORTED = 8,
    DAV1_STATUS_INTERNAL_ERROR = 9,
    DAV1_STATUS_INVALID_STATE = 10,
    DAV1_STATUS_CANCELLED = 11
} dav1_status;

typedef enum dav1_encoder {
    DAV1_ENCODER_VITS = 0,
    DAV1_ENCODER_VITB = 1,
    DAV1_ENCODER_VITL = 2
} dav1_encoder;

typedef struct dav1_create_options {
    uint32_t struct_size;
    uint32_t abi_version;
    dav1_encoder encoder;
    int32_t vulkan_device_index;
    uint32_t flags;
} dav1_create_options;

typedef struct dav1_image_shape {
    int32_t width;
    int32_t height;
} dav1_image_shape;

enum {
    DAV1_GPU_CAP_D3D12_SHARED_INPUT = 1ull << 0u,
    DAV1_GPU_CAP_D3D12_FENCE_WAIT = 1ull << 1u,
    DAV1_GPU_CAP_D3D12_SHARED_OUTPUT = 1ull << 2u,
    DAV1_GPU_CAP_D3D12_FENCE_SIGNAL = 1ull << 3u,
    DAV1_GPU_CAP_ASYNC_SUBMIT = 1ull << 4u,
    DAV1_GPU_CAP_CANCELLATION = 1ull << 5u,
    DAV1_GPU_CAP_NO_HOST_PIXEL_STAGING = 1ull << 6u,
    DAV1_GPU_CAP_NO_HOST_DEPTH_STAGING = 1ull << 7u,
    DAV1_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT = 1ull << 8u,
    DAV1_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT = 1ull << 9u
};

typedef enum dav1_gpu_pixel_format {
    DAV1_GPU_PIXEL_BGRA8 = 1,
    DAV1_GPU_PIXEL_RGBA8 = 2,
    DAV1_GPU_PIXEL_DEPTH_FLOAT32 = 3
} dav1_gpu_pixel_format;

typedef enum dav1_gpu_job_state {
    DAV1_GPU_JOB_QUEUED = 0,
    DAV1_GPU_JOB_RUNNING = 1,
    DAV1_GPU_JOB_COMPLETE = 2,
    DAV1_GPU_JOB_FAILED = 3,
    DAV1_GPU_JOB_CANCELLED = 4
} dav1_gpu_job_state;

typedef struct dav1_gpu_capabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    uint64_t adapter_luid;
    uint32_t maximum_in_flight_jobs;
    uint32_t reserved;
} dav1_gpu_capabilities;

/*
 * Windows-only GPU submission contract.
 *
 * shared_resource_handle is a borrowed NT handle for a shared D3D12 buffer
 * containing tightly addressable BGRA8 or RGBA8 pixels. wait_fence_handle is
 * a borrowed NT handle for a shared D3D12 fence. DAV1 duplicates both handles
 * before returning from submit, so the caller may close them immediately.
 */
typedef struct dav1_d3d12_submit_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t shared_resource_handle;
    uint64_t resource_byte_size;
    uint32_t width;
    uint32_t height;
    uint32_t row_stride_bytes;
    uint32_t pixel_format;
    int32_t input_size;
    uint32_t reserved;
    uint64_t wait_fence_handle;
    uint64_t wait_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dav1_d3d12_submit_request;

/*
 * shared_texture_handle is a borrowed NT handle for a shared D3D12
 * TEXTURE2D. BGRA8 maps to DXGI_FORMAT_B8G8R8A8_UNORM and RGBA8 maps to
 * DXGI_FORMAT_R8G8B8A8_UNORM. DAV1 duplicates/imports the texture and fence
 * handles before returning.
 */
typedef struct dav1_d3d12_texture_submit_request {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t shared_texture_handle;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int32_t input_size;
    uint64_t wait_fence_handle;
    uint64_t wait_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dav1_d3d12_texture_submit_request;

typedef struct dav1_gpu_job_status {
    uint32_t struct_size;
    uint32_t state;
    uint32_t output_count;
    uint32_t reserved;
    uint64_t source_frame_id;
} dav1_gpu_job_status;

/*
 * Handles in this descriptor are borrowed and remain valid until the matching
 * output lease is released. The resource is a D3D12 shared buffer containing
 * row-major float32 depth. Consumers must wait for ready_fence_value on the
 * shared D3D12 fence before accessing it. If a consumer transitions the
 * resource, it must return it to D3D12_RESOURCE_STATE_COMMON before releasing
 * the lease so DAV1 may safely reuse the slot.
 */
typedef struct dav1_d3d12_output_descriptor {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t pixel_format;
    uint32_t reserved;
    uint32_t width;
    uint32_t height;
    uint32_t row_stride_bytes;
    uint32_t reserved2;
    uint64_t byte_size;
    uint64_t shared_resource_handle;
    uint64_t ready_fence_handle;
    uint64_t ready_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dav1_d3d12_output_descriptor;

/*
 * The leased resource is a shared D3D12 TEXTURE2D with
 * DXGI_FORMAT_R32_FLOAT. Both handles are borrowed for the lease lifetime.
 * A consumer must return the texture to D3D12_RESOURCE_STATE_COMMON before
 * releasing the lease if it performs an explicit state transition.
 */
typedef struct dav1_d3d12_texture_output_descriptor {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t pixel_format;
    uint32_t reserved;
    uint32_t width;
    uint32_t height;
    uint32_t reserved2;
    uint32_t reserved3;
    uint64_t shared_texture_handle;
    uint64_t ready_fence_handle;
    uint64_t ready_fence_value;
    uint64_t source_frame_id;
    uint64_t timestamp_ns;
} dav1_d3d12_texture_output_descriptor;

typedef struct dav1_transfer_counters {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t tensor_upload_bytes;
    uint64_t tensor_download_bytes;
} dav1_transfer_counters;

DAV1_API uint32_t DAV1_CALL dav1_abi_version(void);
DAV1_API const char* DAV1_CALL dav1_version_string(void);
DAV1_API const char* DAV1_CALL dav1_status_string(dav1_status status);
DAV1_API const char* DAV1_CALL dav1_last_error(void);

DAV1_API dav1_status DAV1_CALL dav1_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dav1_image_shape* network_shape);

DAV1_API dav1_status DAV1_CALL dav1_create(
    const char* model_path_utf8,
    const dav1_create_options* options,
    dav1_context** context);

DAV1_API void DAV1_CALL dav1_destroy(dav1_context* context);

DAV1_API dav1_status DAV1_CALL dav1_probe_gpu_capabilities(
    int32_t vulkan_device_index,
    dav1_gpu_capabilities* capabilities);

DAV1_API dav1_status DAV1_CALL dav1_get_gpu_capabilities(
    const dav1_context* context,
    dav1_gpu_capabilities* capabilities);

DAV1_API dav1_status DAV1_CALL dav1_get_transfer_counters(
    const dav1_context* context,
    dav1_transfer_counters* counters);

DAV1_API dav1_status DAV1_CALL dav1_submit_d3d12(
    dav1_context* context,
    const dav1_d3d12_submit_request* request,
    dav1_gpu_job** job);

DAV1_API dav1_status DAV1_CALL dav1_submit_d3d12_texture(
    dav1_context* context,
    const dav1_d3d12_texture_submit_request* request,
    dav1_gpu_job** job);

DAV1_API dav1_status DAV1_CALL dav1_gpu_job_poll(
    const dav1_gpu_job* job,
    dav1_gpu_job_status* status);

DAV1_API dav1_status DAV1_CALL dav1_gpu_job_cancel(dav1_gpu_job* job);
DAV1_API void DAV1_CALL dav1_gpu_job_release(dav1_gpu_job* job);

DAV1_API dav1_status DAV1_CALL dav1_gpu_output_acquire(
    dav1_gpu_job* job,
    uint32_t output_index,
    dav1_d3d12_output_descriptor* descriptor,
    dav1_gpu_output_lease** lease);

DAV1_API dav1_status DAV1_CALL dav1_gpu_texture_output_acquire(
    dav1_gpu_job* job,
    uint32_t output_index,
    dav1_d3d12_texture_output_descriptor* descriptor,
    dav1_gpu_output_lease** lease);

DAV1_API void DAV1_CALL dav1_gpu_output_release(
    dav1_gpu_output_lease* lease);

/*
 * Runs the complete Python-compatible image path.
 *
 * Input is interleaved BGR uint8 data, matching cv2.imread. The output is one
 * row-major float32 depth value per source pixel. output_count must be at least
 * image_width * image_height.
 */
DAV1_API dav1_status DAV1_CALL dav1_infer_bgr8(
    dav1_context* context,
    const uint8_t* bgr,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    float* output_depth,
    size_t output_count);

/*
 * Runs only the neural network. Input is normalized RGB planar float32 NCHW
 * data for a batch of one. Width and height must be positive multiples of 14.
 * Output has network_width * network_height float32 values.
 */
DAV1_API dav1_status DAV1_CALL dav1_infer_tensor_f32(
    dav1_context* context,
    const float* normalized_rgb_chw,
    int32_t network_width,
    int32_t network_height,
    float* output_depth,
    size_t output_count);

#ifdef __cplusplus
}
#endif

#endif
