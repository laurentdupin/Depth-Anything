# Native validation status

Reference source revision:
`1d03336771fe09c5398ffdd211441e33941a97dc`.

## ViT-S

- Canonical repository: `LiheYoung/depth_anything_vits14`
- Canonical revision: `1bee667edbfec7f9b345abaffec8c9c2bd5293c2`
- Canonical `pytorch_model.bin` SHA-256:
  `ba0ab5365610467de4fea1ffe64b4ae3a756561a4df9e11bd7369205241f2082`
- Native converter: `dav1-export-pytorch-v1`
- Native format: `DAV1MOD` version 1

On 20 repository images at inference size 140, compared with PyTorch 2.7
CPU float32:

- relative L1 minimum: 0.044614%
- relative L1 median: 0.229834%
- relative L1 maximum: 0.779531%
- maximum observed absolute difference: 0.236098289

The Windows D3D12 shared-texture canary passed on the Radeon RX 9070 for:

- shared BGRA and RGBA texture input;
- the full encoder and DPT graph;
- shared R32_FLOAT texture output;
- output correlation;
- three retained output leases and bounded-slot exhaustion;
- stable slot reuse;
- cancellation, retained-fence lifetime, and shutdown.

The test reports a maximum depth-range-normalized correlation difference of
0.00026811%.

## ViT-B

- Canonical repository: `LiheYoung/depth_anything_vitb14`
- Canonical revision: `a7dff65359777209cb99e1361df8a76144f87e49`
- Canonical `pytorch_model.bin` SHA-256:
  `d25d4cf2b50e63198c42e684abb10b9d242714cec0fe3e2eb8f7da41fada25b3`

On the same 20-image size-140 gate against CPU float32:

- relative L1 minimum: 0.035643%
- relative L1 median: 0.146226%
- relative L1 maximum: 0.430813%
- maximum observed absolute difference: 0.172847748

## ViT-L

- Canonical repository: `LiheYoung/depth_anything_vitl14`
- Canonical revision: `973948530e4e4f4afd6d1913f670d9f96071dcaa`
- Canonical `pytorch_model.bin` SHA-256:
  `6fda5199f0043e91e1c6117df00313db57c630d8ed0306070b43da1ecfffbd9d`

The original size-140 gate exposed one 1.530824% outlier. Encoder-stage
probing localized the divergence to the final transformer blocks. The cause
was not generic float32 reduction order: the native shader implemented
tanh-approximate GELU while the pinned PyTorch graph uses exact-erf GELU.
Replacing it with a high-accuracy erf formulation reduced the worst raw
relative L1 across all 22 assets to 0.002397%.

The deployed InferBridge contract was also checked separately, including its
nearest resize, BGR channel interpretation, width/height ordering, min-max
normalization, and uint8 conversion. The worst results were:

| GPU | Raw relative L1 | Normalized relative L1 | Normalized max pixel | uint8 max delta |
|---|---:|---:|---:|---:|
| Radeon RX 9070 | 0.027619% | 0.019658% | 0.038007% | 1 |
| GeForce GTX 1080 | 0.021980% | 0.015664% | 0.030205% | 1 |
| Radeon RX 6700 XT | 0.023355% | 0.016607% | 0.032026% | 1 |

All ViT-L checks now pass the less-than-1% requirement on all three GPUs.
The additive `dav1_get_inferbridge_shape` and
`dav1_inferbridge_bgra8_u8` image ABI was then exercised directly on the
same 22 images. Its final uint8 payload differs from Python CPU by at most
one level; the legacy-nearest float scale edge case is covered by a unit test.

The stable default deliberately keeps model weights, attention scores, and
activations in float32 and uses the scalar-order Vulkan linear/convolution
paths. Reduced-precision and alternate-tile selection are deferred to the
performance phase.

Build/test gates:

- Vulkan build: 5 executable tests passed; the model-dependent D3D12 canary
  passes when `DAV1_VITS_MODEL` names the derived model.
- no-Vulkan build: 2 tests passed.

## Remaining validation

The correctness capability gate passes for ViT-S, ViT-B, and ViT-L. Broader
multi-size performance measurements remain part of the optimization phase.

## 2026-07-31 performance precision gate

The performance path now keeps ViT-S transformer weights in FP32 and uses
tiled packed-FP16 transformer weights for ViT-B/ViT-L. Activations,
attention scores, DPT weights, and accumulation remain FP32. Pointwise DPT
convolutions use tiled GEMM, while 3x3 DPT convolutions batch eight input
channels between workgroup barriers.

The deployed InferBridge ViT-L contract was rerun on all 22 Depth Anything V2
assets against PyTorch CPU at size 140 on the GTX 1080. The worst normalized
absolute deviation was `0.784314%`; all other images were at or below
`0.392157%`. The final uint8 payload differed by at most two levels. This
remains below the requested 1% maximum-deviation ceiling.
