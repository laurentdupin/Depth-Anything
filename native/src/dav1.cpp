#include "depth_anything_v1.h"

#include "executor.h"
#include "image.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct dav1_context {
    std::shared_ptr<dav1::Executor> executor;
    dav1::ImageScratch image_scratch;
    std::vector<float> image_input;
    std::vector<float> image_output;
};

struct dav1_gpu_job {
    std::atomic<uint32_t> references{1};
    std::shared_ptr<dav1::Executor> executor;
    std::unique_ptr<dav1::GpuJob> implementation;
    uint64_t source_frame_id = 0;
};

struct dav1_gpu_output_lease {
    dav1_gpu_job* job = nullptr;
};

namespace {

thread_local std::string last_error;

class ApiError final : public std::runtime_error {
public:
    ApiError(dav1_status status, const char* message)
        : std::runtime_error(message), status_(status) {}
    dav1_status status() const { return status_; }

private:
    dav1_status status_;
};

dav1_status fail(dav1_status status, const char* message) {
    last_error = message ? message : "";
    return status;
}

bool supported_encoder(dav1_encoder encoder) {
    return encoder == DAV1_ENCODER_VITS ||
        encoder == DAV1_ENCODER_VITB ||
        encoder == DAV1_ENCODER_VITL;
}

void retain_gpu_job(dav1_gpu_job* job) {
    job->references.fetch_add(1, std::memory_order_relaxed);
}

void release_gpu_job(dav1_gpu_job* job) {
    if (job != nullptr &&
        job->references.fetch_sub(
            1, std::memory_order_acq_rel) == 1) {
        delete job;
    }
}

template <typename Function>
dav1_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return DAV1_STATUS_OK;
    } catch (const ApiError& error) {
        return fail(error.status(), error.what());
    } catch (const std::bad_alloc&) {
        return fail(DAV1_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& error) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const dav1::GpuSlotsExhausted& error) {
        return fail(DAV1_STATUS_INVALID_STATE, error.what());
    } catch (const std::exception& error) {
        return fail(DAV1_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(DAV1_STATUS_INTERNAL_ERROR, "unknown internal error");
    }
}

}  // namespace

#if !defined(DAV1_WITH_VULKAN) && !defined(DAV1_WITH_METAL)
namespace dav1 {
std::unique_ptr<Executor> create_executor(
    const std::string&,
    dav1_encoder,
    int, std::uint32_t, const std::string&) {
    throw std::runtime_error(
        "this library was built without a GPU executor");
}
GpuCapabilities probe_gpu_capabilities(int) {
    return {};
}
}  // namespace dav1
#endif

extern "C" {

uint32_t DAV1_CALL dav1_abi_version(void) {
    return DAV1_ABI_VERSION;
}

const char* DAV1_CALL dav1_version_string(void) {
    return "0.1.0";
}

const char* DAV1_CALL dav1_status_string(dav1_status status) {
    switch (status) {
        case DAV1_STATUS_OK: return "ok";
        case DAV1_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case DAV1_STATUS_MODEL_IO: return "model I/O error";
        case DAV1_STATUS_MODEL_FORMAT: return "invalid model format";
        case DAV1_STATUS_VULKAN_UNAVAILABLE: return "Vulkan unavailable";
        case DAV1_STATUS_OUT_OF_MEMORY: return "out of memory";
        case DAV1_STATUS_INFERENCE_FAILED: return "inference failed";
        case DAV1_STATUS_BUFFER_TOO_SMALL: return "buffer too small";
        case DAV1_STATUS_UNSUPPORTED: return "unsupported";
        case DAV1_STATUS_INTERNAL_ERROR: return "internal error";
        case DAV1_STATUS_INVALID_STATE: return "invalid state";
        case DAV1_STATUS_CANCELLED: return "cancelled";
        default: return "unknown status";
    }
}

const char* DAV1_CALL dav1_last_error(void) {
    return last_error.c_str();
}

dav1_status DAV1_CALL dav1_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dav1_image_shape* network_shape) {
    if (network_shape == nullptr) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "network_shape is null");
    }
    return protect([&] {
        const dav1::ImageShape shape =
            dav1::network_shape(image_width, image_height, input_size);
        network_shape->width = shape.width;
        network_shape->height = shape.height;
    });
}

dav1_status DAV1_CALL dav1_get_inferbridge_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    dav1_image_shape* output_shape) {
    if (output_shape == nullptr) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "output_shape is null");
    }
    return protect([&] {
        const dav1::ImageShape shape =
            dav1::network_shape(image_width, image_height, input_size);
        output_shape->width = shape.height;
        output_shape->height = shape.width;
    });
}

dav1_status DAV1_CALL dav1_create(
    const char* model_path_utf8,
    const dav1_create_options* options,
    dav1_context** context) {
    if (context == nullptr) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "context is null");
    }
    *context = nullptr;
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0') {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "model_path_utf8 is empty");
    }
    if (options == nullptr || options->struct_size < sizeof(dav1_create_options)) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "invalid create options");
    }
    if (options->abi_version != DAV1_ABI_VERSION) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "ABI version mismatch");
    }
    if (!supported_encoder(options->encoder)) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "unsupported encoder");
    }
    constexpr uint32_t supported_flags = DAV1_CREATE_FORCE_FP16 |
        DAV1_CREATE_FORCE_INT8 | DAV1_CREATE_FORCE_VULKAN |
        DAV1_CREATE_FORCE_METAL;
    if ((options->flags & ~supported_flags) != 0u)
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "unsupported create flags");
    return protect([&] {
        auto result = std::make_unique<dav1_context>();
        result->executor = std::shared_ptr<dav1::Executor>(
            dav1::create_executor(
                model_path_utf8,
                options->encoder,
                options->vulkan_device_index,
                options->flags,
                std::string()));
        *context = result.release();
    });
}

void DAV1_CALL dav1_destroy(dav1_context* context) {
    delete context;
}

dav1_status DAV1_CALL dav1_probe_gpu_capabilities(
    int32_t vulkan_device_index,
    dav1_gpu_capabilities* capabilities) {
    if (capabilities == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "GPU capabilities are null");
    }
    if (capabilities->struct_size < sizeof(*capabilities) ||
        vulkan_device_index < 0) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "invalid GPU capability probe");
    }
    return protect([&] {
        const dav1::GpuCapabilities available =
            dav1::probe_gpu_capabilities(vulkan_device_index);
        *capabilities = {};
        capabilities->struct_size = sizeof(*capabilities);
        capabilities->abi_version = DAV1_ABI_VERSION;
        capabilities->flags = available.flags;
        capabilities->adapter_luid = available.adapter_luid;
        capabilities->maximum_in_flight_jobs =
            available.maximum_in_flight_jobs;
    });
}

dav1_status DAV1_CALL dav1_get_gpu_capabilities(
    const dav1_context* context,
    dav1_gpu_capabilities* capabilities) {
    if (context == nullptr || capabilities == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "null GPU capability argument");
    }
    if (capabilities->struct_size < sizeof(*capabilities)) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "GPU capabilities struct is too small");
    }
    return protect([&] {
        const dav1::GpuCapabilities available =
            context->executor->gpu_capabilities();
        *capabilities = {};
        capabilities->struct_size = sizeof(*capabilities);
        capabilities->abi_version = DAV1_ABI_VERSION;
        capabilities->flags = available.flags;
        capabilities->adapter_luid = available.adapter_luid;
        capabilities->maximum_in_flight_jobs =
            available.maximum_in_flight_jobs;
    });
}

dav1_status DAV1_CALL dav1_get_transfer_counters(
    const dav1_context* context,
    dav1_transfer_counters* counters) {
    if (context == nullptr || counters == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "null transfer counter argument");
    }
    if (counters->struct_size < sizeof(*counters)) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "transfer counters struct is too small");
    }
    return protect([&] {
        std::uint64_t upload = 0;
        std::uint64_t download = 0;
        context->executor->transfer_counters(upload, download);
        *counters = {};
        counters->struct_size = sizeof(*counters);
        counters->tensor_upload_bytes = upload;
        counters->tensor_download_bytes = download;
    });
}

dav1_status DAV1_CALL dav1_submit_d3d12(
    dav1_context* context,
    const dav1_d3d12_submit_request* request,
    dav1_gpu_job** job) {
    if (context == nullptr || request == nullptr || job == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "null D3D12 submit argument");
    }
    *job = nullptr;
    constexpr std::size_t binding_v1_size =
        offsetof(dav1_d3d12_texture_binding_request, input_texture_identity);
    if (request->struct_size < binding_v1_size ||
        request->abi_version != DAV1_ABI_VERSION) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "invalid D3D12 submit request");
    }
    return protect([&] {
        if (context->executor->gpu_capabilities().flags == 0) {
            throw ApiError(
                DAV1_STATUS_UNSUPPORTED,
                "complete D3D12/Vulkan GPU interop is unavailable");
        }
        dav1::GpuSubmitRequest native;
        native.shared_resource_handle =
            static_cast<std::uintptr_t>(
                request->shared_resource_handle);
        native.resource_byte_size = request->resource_byte_size;
        native.width = request->width;
        native.height = request->height;
        native.row_stride_bytes = request->row_stride_bytes;
        native.pixel_format =
            static_cast<dav1_gpu_pixel_format>(
                request->pixel_format);
        native.input_size = request->input_size;
        native.wait_fence_handle =
            static_cast<std::uintptr_t>(
                request->wait_fence_handle);
        native.wait_fence_value = request->wait_fence_value;
        native.source_frame_id = request->source_frame_id;
        native.timestamp_ns = request->timestamp_ns;
        auto result = std::make_unique<dav1_gpu_job>();
        result->executor = context->executor;
        result->source_frame_id = request->source_frame_id;
        result->implementation =
            context->executor->submit_gpu(native);
        *job = result.release();
    });
}

dav1_status DAV1_CALL dav1_submit_d3d12_texture(
    dav1_context* context,
    const dav1_d3d12_texture_submit_request* request,
    dav1_gpu_job** job) {
    if (context == nullptr || request == nullptr || job == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "null D3D12 texture submit argument");
    }
    *job = nullptr;
    if (request->struct_size < sizeof(*request) ||
        request->abi_version != DAV1_ABI_VERSION) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "invalid D3D12 texture submit request");
    }
    return protect([&] {
        const std::uint64_t required =
            DAV1_GPU_CAP_D3D12_SHARED_TEXTURE_INPUT |
            DAV1_GPU_CAP_D3D12_SHARED_TEXTURE_OUTPUT;
        if ((context->executor->gpu_capabilities().flags &
             required) != required) {
            throw ApiError(
                DAV1_STATUS_UNSUPPORTED,
                "complete D3D12/Vulkan texture interop is unavailable");
        }
        dav1::GpuTextureSubmitRequest native;
        native.shared_texture_handle =
            static_cast<std::uintptr_t>(
                request->shared_texture_handle);
        native.width = request->width;
        native.height = request->height;
        native.pixel_format =
            static_cast<dav1_gpu_pixel_format>(
                request->pixel_format);
        native.input_size = request->input_size;
        native.wait_fence_handle =
            static_cast<std::uintptr_t>(
                request->wait_fence_handle);
        native.wait_fence_value = request->wait_fence_value;
        native.source_frame_id = request->source_frame_id;
        native.timestamp_ns = request->timestamp_ns;
        auto result = std::make_unique<dav1_gpu_job>();
        result->executor = context->executor;
        result->source_frame_id = request->source_frame_id;
        result->implementation =
            context->executor->submit_gpu_texture(native);
        *job = result.release();
    });
}

dav1_status DAV1_CALL dav1_submit_d3d12_texture_binding(
    dav1_context* context,
    const dav1_d3d12_texture_binding_request* request,
    dav1_gpu_job** job) {
    if (context == nullptr || request == nullptr || job == nullptr) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "null D3D12 texture binding argument");
    }
    *job = nullptr;
    if (request->struct_size < sizeof(*request) ||
        request->abi_version != DAV1_ABI_VERSION) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "invalid D3D12 texture binding request");
    }
    return protect([&] {
        dav1::GpuTextureSubmitRequest native;
        native.shared_texture_handle = static_cast<std::uintptr_t>(request->input_texture_handle);
        const bool has_stable_identities = request->struct_size >= sizeof(*request);
        native.shared_texture_identity = has_stable_identities &&
                request->input_texture_identity != 0
            ? request->input_texture_identity : request->input_texture_handle;
        native.width = request->input_width;
        native.height = request->input_height;
        native.pixel_format = static_cast<dav1_gpu_pixel_format>(request->input_pixel_format);
        native.input_size = request->input_size;
        native.wait_fence_handle = static_cast<std::uintptr_t>(request->wait_fence_handle);
        native.wait_fence_value = request->wait_fence_value;
        native.output_texture_handle = static_cast<std::uintptr_t>(request->output_texture_handle);
        native.output_texture_identity = has_stable_identities &&
                request->output_texture_identity != 0
            ? request->output_texture_identity : request->output_texture_handle;
        native.output_width = request->output_width;
        native.output_height = request->output_height;
        native.signal_fence_handle = static_cast<std::uintptr_t>(request->signal_fence_handle);
        native.signal_fence_value = request->signal_fence_value;
        native.source_frame_id = request->source_frame_id;
        native.timestamp_ns = request->timestamp_ns;
        auto result = std::make_unique<dav1_gpu_job>();
        result->executor = context->executor;
        result->source_frame_id = request->source_frame_id;
        result->implementation = context->executor->submit_gpu_texture(native);
        *job = result.release();
    });
}

dav1_status DAV1_CALL dav1_submit_metal_texture_binding(
    dav1_context* context,
    const dav1_metal_texture_binding_request* request,
    dav1_gpu_job** job) {
    if (context == nullptr || request == nullptr || job == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "null Metal texture binding argument");
    }
    *job = nullptr;
    if (request->struct_size < sizeof(*request) ||
        request->abi_version != DAV1_ABI_VERSION) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "invalid Metal texture binding request");
    }
    return protect([&] {
        dav1::GpuTextureSubmitRequest native;
        native.shared_texture_handle =
            static_cast<std::uintptr_t>(request->input_texture);
        native.shared_texture_identity = native.shared_texture_handle;
        native.width = request->input_width;
        native.height = request->input_height;
        native.pixel_format = static_cast<dav1_gpu_pixel_format>(
            request->input_pixel_format);
        native.input_size = request->input_size;
        native.wait_fence_handle =
            static_cast<std::uintptr_t>(request->wait_shared_event);
        native.wait_fence_value = request->wait_value;
        native.output_texture_handle =
            static_cast<std::uintptr_t>(request->output_texture);
        native.output_texture_identity = native.output_texture_handle;
        native.output_width = request->output_width;
        native.output_height = request->output_height;
        native.signal_fence_handle =
            static_cast<std::uintptr_t>(request->signal_shared_event);
        native.signal_fence_value = request->signal_value;
        native.source_frame_id = request->source_frame_id;
        native.timestamp_ns = request->timestamp_ns;
        auto result = std::make_unique<dav1_gpu_job>();
        result->executor = context->executor;
        result->source_frame_id = request->source_frame_id;
        result->implementation =
            context->executor->submit_gpu_texture(native);
        *job = result.release();
    });
}

dav1_status DAV1_CALL dav1_gpu_job_poll(
    const dav1_gpu_job* job,
    dav1_gpu_job_status* status) {
    if (job == nullptr || status == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "null GPU job poll argument");
    }
    if (status->struct_size < sizeof(*status)) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "GPU job status struct is too small");
    }
    return protect([&] {
        const dav1_gpu_job_state state =
            job->implementation->state();
        *status = {};
        status->struct_size = sizeof(*status);
        status->state = state;
        status->output_count =
            state == DAV1_GPU_JOB_CANCELLED ? 0u : 1u;
        status->source_frame_id = job->source_frame_id;
    });
}

dav1_status DAV1_CALL dav1_gpu_job_cancel(dav1_gpu_job* job) {
    if (job == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "GPU job is null");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAV1_GPU_JOB_COMPLETE) {
            throw ApiError(
                DAV1_STATUS_INVALID_STATE,
                "completed GPU job cannot be cancelled");
        }
        job->implementation->cancel();
    });
}

void DAV1_CALL dav1_gpu_job_release(dav1_gpu_job* job) {
    release_gpu_job(job);
}

dav1_status DAV1_CALL dav1_gpu_output_acquire(
    dav1_gpu_job* job,
    uint32_t output_index,
    dav1_d3d12_output_descriptor* descriptor,
    dav1_gpu_output_lease** lease) {
    if (job == nullptr || descriptor == nullptr || lease == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "null GPU output acquire argument");
    }
    *lease = nullptr;
    if (descriptor->struct_size < sizeof(*descriptor)) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "GPU output descriptor struct is too small");
    }
    if (output_index != 0) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "GPU output index does not exist");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAV1_GPU_JOB_CANCELLED) {
            throw ApiError(
                DAV1_STATUS_CANCELLED,
                "cancelled GPU job has no output");
        }
        const dav1::GpuOutput output =
            job->implementation->output();
        if (output.kind != dav1::GpuOutputKind::buffer) {
            throw ApiError(
                DAV1_STATUS_UNSUPPORTED,
                "GPU job output is a D3D12 texture");
        }
        auto result =
            std::make_unique<dav1_gpu_output_lease>();
        retain_gpu_job(job);
        result->job = job;
        *descriptor = {};
        descriptor->struct_size = sizeof(*descriptor);
        descriptor->abi_version = DAV1_ABI_VERSION;
        descriptor->pixel_format =
            DAV1_GPU_PIXEL_DEPTH_FLOAT32;
        descriptor->width = output.width;
        descriptor->height = output.height;
        descriptor->row_stride_bytes =
            output.row_stride_bytes;
        descriptor->byte_size = output.byte_size;
        descriptor->shared_resource_handle =
            output.shared_resource_handle;
        descriptor->ready_fence_handle =
            output.ready_fence_handle;
        descriptor->ready_fence_value =
            output.ready_fence_value;
        descriptor->source_frame_id =
            output.source_frame_id;
        descriptor->timestamp_ns = output.timestamp_ns;
        *lease = result.release();
    });
}

dav1_status DAV1_CALL dav1_gpu_texture_output_acquire(
    dav1_gpu_job* job,
    uint32_t output_index,
    dav1_d3d12_texture_output_descriptor* descriptor,
    dav1_gpu_output_lease** lease) {
    if (job == nullptr || descriptor == nullptr || lease == nullptr) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "null GPU texture output acquire argument");
    }
    *lease = nullptr;
    if (descriptor->struct_size < sizeof(*descriptor)) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "GPU texture output descriptor struct is too small");
    }
    if (output_index != 0) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "GPU texture output index does not exist");
    }
    return protect([&] {
        if (job->implementation->state() ==
            DAV1_GPU_JOB_CANCELLED) {
            throw ApiError(
                DAV1_STATUS_CANCELLED,
                "cancelled GPU job has no texture output");
        }
        const dav1::GpuOutput output =
            job->implementation->output();
        if (output.kind != dav1::GpuOutputKind::texture) {
            throw ApiError(
                DAV1_STATUS_UNSUPPORTED,
                "GPU job output is a D3D12 buffer");
        }
        auto result =
            std::make_unique<dav1_gpu_output_lease>();
        retain_gpu_job(job);
        result->job = job;
        *descriptor = {};
        descriptor->struct_size = sizeof(*descriptor);
        descriptor->abi_version = DAV1_ABI_VERSION;
        descriptor->pixel_format =
            DAV1_GPU_PIXEL_DEPTH_FLOAT32;
        descriptor->width = output.width;
        descriptor->height = output.height;
        descriptor->shared_texture_handle =
            output.shared_resource_handle;
        descriptor->ready_fence_handle =
            output.ready_fence_handle;
        descriptor->ready_fence_value =
            output.ready_fence_value;
        descriptor->source_frame_id =
            output.source_frame_id;
        descriptor->timestamp_ns = output.timestamp_ns;
        *lease = result.release();
    });
}

void DAV1_CALL dav1_gpu_output_release(
    dav1_gpu_output_lease* lease) {
    if (lease == nullptr) return;
    release_gpu_job(lease->job);
    delete lease;
}

dav1_status DAV1_CALL dav1_infer_tensor_f32(
    dav1_context* context,
    const float* normalized_rgb_chw,
    int32_t network_width,
    int32_t network_height,
    float* output_depth,
    size_t output_count) {
    if (context == nullptr || normalized_rgb_chw == nullptr || output_depth == nullptr) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "null inference argument");
    }
    if (network_width <= 0 || network_height <= 0 ||
        network_width % 14 != 0 || network_height % 14 != 0) {
        return fail(
            DAV1_STATUS_INVALID_ARGUMENT,
            "network dimensions must be positive multiples of 14");
    }
    const size_t required =
        static_cast<size_t>(network_width) * static_cast<size_t>(network_height);
    if (output_count < required) {
        return fail(DAV1_STATUS_BUFFER_TOO_SMALL, "output buffer is too small");
    }
    return protect([&] {
        context->executor->infer(
            normalized_rgb_chw, network_width, network_height, output_depth);
    });
}

dav1_status DAV1_CALL dav1_infer_bgr8(
    dav1_context* context,
    const uint8_t* bgr,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    float* output_depth,
    size_t output_count) {
    if (context == nullptr || bgr == nullptr || output_depth == nullptr) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "null inference argument");
    }
    const size_t required =
        image_width > 0 && image_height > 0
        ? static_cast<size_t>(image_width) * static_cast<size_t>(image_height)
        : 0;
    if (output_count < required) {
        return fail(DAV1_STATUS_BUFFER_TOO_SMALL, "output buffer is too small");
    }
    return protect([&] {
        const dav1::ImageShape shape =
            dav1::network_shape(image_width, image_height, input_size);
        dav1::preprocess_bgr8(
            bgr,
            image_width,
            image_height,
            row_stride_bytes,
            shape,
            context->image_scratch,
            context->image_input);
        context->executor->infer_resized(
            context->image_input.data(),
            shape.width,
            shape.height,
            output_depth,
            image_width,
            image_height);
    });
}

dav1_status DAV1_CALL dav1_inferbridge_bgra8_u8(
    dav1_context* context,
    const uint8_t* bgra,
    int32_t image_width,
    int32_t image_height,
    ptrdiff_t row_stride_bytes,
    int32_t input_size,
    uint8_t* output_depth,
    size_t output_count) {
    if (context == nullptr || bgra == nullptr || output_depth == nullptr) {
        return fail(DAV1_STATUS_INVALID_ARGUMENT, "null inference argument");
    }
    return protect([&] {
        const dav1::ImageShape upstream =
            dav1::network_shape(image_width, image_height, input_size);
        const dav1::ImageShape output_shape{
            upstream.height, upstream.width};
        const size_t required =
            static_cast<size_t>(output_shape.width) * output_shape.height;
        if (output_count < required) {
            throw ApiError(
                DAV1_STATUS_BUFFER_TOO_SMALL,
                "output buffer is too small");
        }
        dav1::preprocess_inferbridge_bgra8(
            bgra,
            image_width,
            image_height,
            row_stride_bytes,
            output_shape,
            context->image_input);
        context->image_output.resize(required);
        context->executor->infer(
            context->image_input.data(),
            output_shape.width,
            output_shape.height,
            context->image_output.data());
        const auto bounds = std::minmax_element(
            context->image_output.begin(), context->image_output.end());
        const float minimum = *bounds.first;
        const float span = *bounds.second - minimum;
        if (!(span > 0.0f)) {
            std::fill_n(output_depth, required, uint8_t{0});
            return;
        }
        for (size_t index = 0; index < required; ++index) {
            const float normalized =
                (context->image_output[index] - minimum) / span * 255.0f;
            output_depth[index] = static_cast<uint8_t>(
                std::clamp(normalized, 0.0f, 255.0f));
        }
    });
}

}  // extern "C"

#include "linux_capture.inl"
