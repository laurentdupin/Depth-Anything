"""Compare pinned PyTorch CPU and native Vulkan DINOv2 feature taps."""

from __future__ import annotations

import argparse
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


torchvision = types.ModuleType("torchvision")
torchvision_transforms = types.ModuleType("torchvision.transforms")
torchvision_transforms.Compose = Compose
torchvision.transforms = torchvision_transforms
sys.modules["torchvision"] = torchvision
sys.modules["torchvision.transforms"] = torchvision_transforms

repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))

from depth_anything.dpt import DepthAnything  # noqa: E402
from depth_anything.util.transform import (  # noqa: E402
    NormalizeImage,
    PrepareForNet,
    Resize,
)


CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def prepared_image(path: Path, input_size: int) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    transform = Compose(
        [
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            PrepareForNet(),
        ]
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    return torch.from_numpy(
        transform({"image": rgb})["image"]
    ).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=CONFIGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=140)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    model = DepthAnything(CONFIGS[args.encoder])
    model.load_state_dict(state)
    model.eval()
    tensor = prepared_image(args.image, args.input_size)
    height, width = tensor.shape[-2:]

    with torch.inference_mode():
        prepared = model.pretrained.prepare_tokens_with_masks(tensor)[0]
        tapped = model.pretrained.get_intermediate_layers(
            tensor, 4, return_class_token=True
        )
    reference = [
        torch.cat((class_token[:, None], patches), dim=1)[0]
        .contiguous()
        .numpy()
        for patches, class_token in tapped
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.output / "input.f32"
    prefix = args.output / "native"
    tensor[0].contiguous().numpy().tofile(input_path)
    subprocess.run(
        [
            str(args.probe.resolve()),
            str(args.model.resolve()),
            args.encoder,
            str(width),
            str(height),
            str(input_path),
            str(prefix),
        ],
        check=True,
    )

    native_prepared = np.fromfile(
        Path(f"{prefix}.tokens.f32"), dtype=np.float32
    ).reshape(prepared.shape)
    prepared_difference = np.abs(
        native_prepared - prepared.contiguous().numpy()
    )
    prepared_relative_l1 = float(
        prepared_difference.astype(np.float64).sum()
        / np.abs(prepared.numpy(), dtype=np.float64).sum()
    )
    print(
        "tokens: "
        f"relative_l1={prepared_relative_l1 * 100:.9f}% "
        f"max_abs={float(prepared_difference.max()):.9g}"
    )

    for index, expected in enumerate(reference):
        actual = np.fromfile(
            Path(f"{prefix}.{index}.f32"), dtype=np.float32
        ).reshape(expected.shape)
        difference = np.abs(actual - expected)
        relative_l1 = float(
            difference.astype(np.float64).sum()
            / np.abs(expected, dtype=np.float64).sum()
        )
        print(
            f"tap {index}: relative_l1={relative_l1 * 100:.9f}% "
            f"max_abs={float(difference.max()):.9g}"
        )


if __name__ == "__main__":
    main()
