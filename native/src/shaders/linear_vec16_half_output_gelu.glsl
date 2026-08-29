#version 450 core

#define K_VECTORS 8
#define WIDE_ROW_TILE
#define HALF_WEIGHT
#define HALF_OUTPUT
#define FUSED_GELU
#include "linear_vec4_common.glsl"
