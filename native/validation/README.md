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
0.043247%.

Build/test gates:

- Vulkan build: 5 executable tests passed; the model-dependent D3D12 canary
  passes when `DAV1_VITS_MODEL` names the derived model.
- no-Vulkan build: 2 tests passed.

## Pending

ViT-B and ViT-L canonical checkpoints and all size variants still require the
same end-to-end accuracy gate. Native capability must not be advertised for
those variants before it passes.
