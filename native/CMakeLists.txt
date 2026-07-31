cmake_minimum_required(VERSION 3.22)

project(depth_anything_v1_c VERSION 0.1.0 LANGUAGES C CXX)

if(MSVC)
    # The distributable must be one DLL, so do not require the separately
    # installed Visual C++ runtime DLLs.
    set(
        CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
endif()

option(DAV1_BUILD_TESTS "Build the C ABI and preprocessing tests" ON)
option(DAV1_WITH_VULKAN "Build the dependency-free Vulkan executor" ON)
option(DAV1_BUILD_TOOLS "Build native command-line tools" ON)

if(DAV1_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)

    set(DAV1_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${DAV1_GENERATED_DIR}")

    function(dav1_compile_shader NAME)
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${DAV1_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${DAV1_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}"
                "-DVARIABLE=dav1_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM
        )
        set(DAV1_SHADER_HEADERS ${DAV1_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()

    dav1_compile_shader(elementwise)
    dav1_compile_shader(linear)
    dav1_compile_shader(linear16)
    dav1_compile_shader(linear_half)
    dav1_compile_shader(linear16_half)
    dav1_compile_shader(linear_vec8)
    dav1_compile_shader(linear_vec8_half)
    dav1_compile_shader(gelu)
    dav1_compile_shader(layer_norm)
    dav1_compile_shader(add_scaled)
    dav1_compile_shader(prepare_tokens)
    dav1_compile_shader(position_bicubic)
    dav1_compile_shader(add_position)
    dav1_compile_shader(add)
    dav1_compile_shader(bmm)
    dav1_compile_shader(bmm_score_half)
    dav1_compile_shader(bmm_value_half)
    dav1_compile_shader(softmax_lastdim)
    dav1_compile_shader(softmax_lastdim_half)
    dav1_compile_shader(project_tokens)
    dav1_compile_shader(project_tokens_half)
    dav1_compile_shader(conv2d)
    dav1_compile_shader(conv2d_pointwise_gemm)
    dav1_compile_shader(conv2d8)
    dav1_compile_shader(conv2d_half)
    dav1_compile_shader(conv2d8_half)
    dav1_compile_shader(conv2d_tiled)
    dav1_compile_shader(conv2d_tiled16x8)
    dav1_compile_shader(conv_transpose_nonoverlap)
    dav1_compile_shader(conv_transpose_nonoverlap_half)
    dav1_compile_shader(bilinear_align_false)
    dav1_compile_shader(bilinear_align_false_image)
    dav1_compile_shader(bilinear_align_true)
    dav1_compile_shader(bilinear_align_true_image)
    dav1_compile_shader(relu)
    dav1_compile_shader(preprocess_rgba)
    dav1_compile_shader(preprocess_texture)
endif()

add_library(depth_anything_v1 SHARED
    src/dav1.cpp
    src/image.cpp
    src/inferbridge_harness.cpp
)
add_library(depth_anything_v1::depth_anything_v1 ALIAS depth_anything_v1)

target_compile_features(depth_anything_v1 PRIVATE cxx_std_17)
target_include_directories(depth_anything_v1
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_definitions(depth_anything_v1
    PRIVATE
        DAV1_BUILD_DLL
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY
        $<$<BOOL:${DAV1_WITH_VULKAN}>:DAV1_WITH_VULKAN=1>
)
set_target_properties(depth_anything_v1 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(DAV1_WITH_VULKAN)
    target_sources(depth_anything_v1 PRIVATE
        src/model.cpp
        src/gpu_model.cpp
        src/encoder.cpp
        src/dpt.cpp
        src/operators.cpp
        src/vulkan.cpp
        src/gpu_preprocess.cpp
        src/vulkan_executor.cpp
        ${DAV1_SHADER_HEADERS}
    )
    target_include_directories(depth_anything_v1 PRIVATE "${DAV1_GENERATED_DIR}")
    target_link_libraries(depth_anything_v1 PRIVATE Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(depth_anything_v1 PRIVATE d3d12 dxgi)
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS depth_anything_v1
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES include/depth_anything_v1.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES include/inferbridge_harness.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

if(DAV1_BUILD_TOOLS)
    add_executable(dav1_run tools/dav1_run.c)
    target_link_libraries(dav1_run PRIVATE depth_anything_v1)
    install(TARGETS dav1_run RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

    if(DAV1_WITH_VULKAN)
        add_executable(dav1_encoder_probe
            tools/dav1_encoder_probe.cpp
            src/encoder.cpp
            src/dpt.cpp
            src/gpu_model.cpp
            src/model.cpp
            src/operators.cpp
            src/vulkan.cpp
            src/gpu_preprocess.cpp
            ${DAV1_SHADER_HEADERS}
        )
        target_compile_features(dav1_encoder_probe PRIVATE cxx_std_17)
        target_include_directories(dav1_encoder_probe PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DAV1_GENERATED_DIR}"
        )
        target_link_libraries(dav1_encoder_probe PRIVATE Vulkan::Vulkan)
    endif()
endif()

if(DAV1_BUILD_TESTS)
    enable_testing()
    add_executable(dav1_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(dav1_c_abi_smoke PRIVATE depth_anything_v1)
    add_test(NAME dav1_c_abi_smoke COMMAND dav1_c_abi_smoke)

    add_executable(dav1_harness_abi_smoke tests/harness_abi_smoke.c)
    target_link_libraries(dav1_harness_abi_smoke PRIVATE depth_anything_v1)
    add_test(NAME dav1_harness_abi_smoke COMMAND dav1_harness_abi_smoke)

    add_executable(dav1_harness_full_graph tests/harness_full_graph.cpp)
    target_compile_features(dav1_harness_full_graph PRIVATE cxx_std_17)
    target_link_libraries(dav1_harness_full_graph PRIVATE depth_anything_v1)
    add_test(NAME dav1_harness_full_graph COMMAND dav1_harness_full_graph)
    set_tests_properties(
        dav1_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(dav1_image_tests tests/image_tests.cpp src/image.cpp)
    target_include_directories(dav1_image_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME dav1_image_tests COMMAND dav1_image_tests)

    add_executable(dav1_preprocess_probe tools/dav1_preprocess_probe.cpp src/image.cpp)
    target_include_directories(
        dav1_preprocess_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

    if(DAV1_WITH_VULKAN)
        add_executable(dav1_vulkan_smoke
            tests/vulkan_smoke.cpp
            src/gpu_model.cpp
            src/model.cpp
            src/operators.cpp
            src/vulkan.cpp
            src/gpu_preprocess.cpp
            ${DAV1_SHADER_HEADERS}
        )
        target_compile_features(dav1_vulkan_smoke PRIVATE cxx_std_17)
        target_include_directories(dav1_vulkan_smoke PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DAV1_GENERATED_DIR}"
        )
        target_link_libraries(dav1_vulkan_smoke PRIVATE Vulkan::Vulkan)
        add_test(NAME dav1_vulkan_smoke COMMAND dav1_vulkan_smoke)

        if(WIN32)
            add_executable(
                dav1_model_metadata_tests
                tests/model_metadata.cpp
                src/model.cpp)
            target_compile_features(
                dav1_model_metadata_tests PRIVATE cxx_std_17)
            target_include_directories(
                dav1_model_metadata_tests PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src")
            add_test(
                NAME dav1_model_metadata_tests
                COMMAND dav1_model_metadata_tests)

            add_executable(dav1_vulkan_external_smoke
                tests/vulkan_external_smoke.cpp
                src/gpu_preprocess.cpp
                src/image.cpp
                src/vulkan.cpp
                ${DAV1_SHADER_HEADERS}
            )
            target_compile_features(
                dav1_vulkan_external_smoke PRIVATE cxx_std_17)
            target_include_directories(
                dav1_vulkan_external_smoke PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${DAV1_GENERATED_DIR}"
            )
            target_link_libraries(
                dav1_vulkan_external_smoke PRIVATE
                    Vulkan::Vulkan d3d12 dxgi)
            add_test(
                NAME dav1_vulkan_external_smoke
                COMMAND dav1_vulkan_external_smoke)
            set_tests_properties(
                dav1_vulkan_external_smoke
                PROPERTIES SKIP_RETURN_CODE 77)

            add_executable(
                dav1_d3d12_full_graph
                tests/d3d12_full_graph.cpp)
            target_compile_features(
                dav1_d3d12_full_graph PRIVATE cxx_std_17)
            target_link_libraries(
                dav1_d3d12_full_graph PRIVATE
                    depth_anything_v1 d3d12 dxgi)
            target_compile_definitions(
                dav1_d3d12_full_graph PRIVATE
                    "DAV1_DEFAULT_VITS_MODEL=\"${CMAKE_CURRENT_SOURCE_DIR}/validation/depth_anything_v1_vits.dav1\"")
            add_test(
                NAME dav1_d3d12_full_graph
                COMMAND dav1_d3d12_full_graph)
            set_tests_properties(
                dav1_d3d12_full_graph
                PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
