#include "depth_anything_v1.h"

#include <assert.h>
#include <string.h>

int main(void) {
    assert(dav1_abi_version() == DAV1_ABI_VERSION);
    assert(strlen(dav1_version_string()) != 0);
    assert(strcmp(dav1_status_string(DAV1_STATUS_OK), "ok") == 0);
    assert(
        strcmp(
            dav1_status_string(DAV1_STATUS_CANCELLED),
            "cancelled") == 0);

    dav1_image_shape shape = {0, 0};
    assert(dav1_get_network_shape(640, 480, 518, &shape) == DAV1_STATUS_OK);
    assert(shape.width == 686);
    assert(shape.height == 518);
    assert(dav1_get_network_shape(480, 640, 518, &shape) == DAV1_STATUS_OK);
    assert(shape.width == 518);
    assert(shape.height == 686);
    assert(
        dav1_get_inferbridge_shape(480, 640, 518, &shape) ==
        DAV1_STATUS_OK);
    assert(shape.width == 686);
    assert(shape.height == 518);

    assert(
        dav1_get_network_shape(0, 480, 518, &shape) ==
        DAV1_STATUS_INVALID_ARGUMENT);
    assert(strlen(dav1_last_error()) != 0);
    dav1_gpu_capabilities capabilities = {0};
    capabilities.struct_size = sizeof(capabilities);
    assert(
        dav1_probe_gpu_capabilities(0, &capabilities) ==
        DAV1_STATUS_OK);
    assert(capabilities.abi_version == DAV1_ABI_VERSION);
    capabilities.struct_size = sizeof(capabilities);
    assert(
        dav1_get_gpu_capabilities(NULL, &capabilities) ==
        DAV1_STATUS_INVALID_ARGUMENT);
    dav1_d3d12_texture_submit_request texture_request = {0};
    texture_request.struct_size = sizeof(texture_request);
    texture_request.abi_version = DAV1_ABI_VERSION;
    dav1_gpu_job* job = NULL;
    assert(
        dav1_submit_d3d12_texture(
            NULL, &texture_request, &job) ==
        DAV1_STATUS_INVALID_ARGUMENT);
    return 0;
}
