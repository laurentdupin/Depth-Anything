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

The size-140 gate currently has relative L1 minimum 0.040307%, median
0.206830%, and maximum 1.530824%. Nineteen images are below 0.373%; one image
is the outlier. ViT-L therefore remains pending the less-than-1% maximum
accuracy gate even though the difference is consistent with accumulated
float32 execution-order divergence.

The stable default deliberately keeps model weights, attention scores, and
activations in float32 and uses the scalar-order Vulkan linear/convolution
paths. Reduced-precision and alternate-tile selection are deferred to the
performance phase.

Build/test gates:

- Vulkan build: 5 executable tests passed; the model-dependent D3D12 canary
  passes when `DAV1_VITS_MODEL` names the derived model.
- no-Vulkan build: 2 tests passed.

## Pending

ViT-L and the multi-size matrix still require the same end-to-end accuracy
gate. Native capability must not be advertised for a variant before it passes.
