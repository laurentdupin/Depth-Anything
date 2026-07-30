"""Compare the native DLL with the Python Vulkan reference on repository images.

This is a development validation tool. Python, PyTorch, OpenCV, and NumPy are
not used by or shipped with the native runtime.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import sys
import time
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


# dpt.py imports torchvision only for this trivial composition helper. Avoid
# requiring a functioning TorchVision installation in the validation setup.
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
        "enum": 0,
        "model": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
    },
    "vitb": {
        "enum": 1,
        "model": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
    },
    "vitl": {
        "enum": 2,
        "model": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    },
}


def load_reference_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
    except RuntimeError as error:
        if "TorchScript archives" not in str(error):
            raise
    scripted = torch.jit.load(str(checkpoint), map_location="cpu")
    state = {}
    encoder_prefixes = (
        "cls_token",
        "patch_embed.",
        "blocks.",
        "norm.",
    )
    for name, value in scripted.state_dict().items():
        if name == "base_pos_embed":
            mapped = "pretrained.pos_embed"
        elif name.startswith(encoder_prefixes):
            mapped = "pretrained." + name
        else:
            mapped = name
        state[mapped] = value
    return state


class CreateOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("encoder", ctypes.c_int),
        ("vulkan_device_index", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class ImageShape(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
    ]


def configure_dll(path: Path):
    dll = ctypes.CDLL(str(path.resolve()))
    dll.dav1_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(CreateOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    dll.dav1_create.restype = ctypes.c_int
    dll.dav1_destroy.argtypes = [ctypes.c_void_p]
    dll.dav1_infer_bgr8.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_ssize_t,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    dll.dav1_infer_bgr8.restype = ctypes.c_int
    dll.dav1_infer_tensor_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    dll.dav1_infer_tensor_f32.restype = ctypes.c_int
    dll.dav1_get_inferbridge_shape.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ImageShape),
    ]
    dll.dav1_get_inferbridge_shape.restype = ctypes.c_int
    dll.dav1_inferbridge_bgra8_u8.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_ssize_t,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    dll.dav1_inferbridge_bgra8_u8.restype = ctypes.c_int
    dll.dav1_last_error.restype = ctypes.c_char_p
    return dll


def python_input(
    image: np.ndarray, input_size: int, device: str
) -> torch.Tensor:
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
    prepared = transform({"image": rgb})["image"]
    return torch.from_numpy(prepared).unsqueeze(0).to(device)


def inferbridge_input(
    image: np.ndarray, input_size: int, device: str
) -> torch.Tensor:
    resize = Resize(
        width=input_size,
        height=input_size,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_NEAREST,
    )
    target_width, target_height = resize.get_size(
        image.shape[1], image.shape[0]
    )
    tensor = torch.from_numpy(image).to(device)
    tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)
    # Preserve the deployed InferBridge template's (width, height) ordering
    # and its BGR channel interpretation.
    tensor = torch.nn.functional.interpolate(
        tensor, (target_width, target_height)
    ).float()
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)
    return (tensor / 255.0 - mean) / std


def normalized_metrics(
    native: np.ndarray, reference: np.ndarray
) -> dict[str, float | int]:
    def normalize(value: np.ndarray) -> np.ndarray:
        minimum = float(value.min())
        span = float(value.max()) - minimum
        if span == 0.0:
            return np.zeros_like(value, dtype=np.float32)
        return (value.astype(np.float32, copy=False) - minimum) / span

    native_normalized = normalize(native)
    reference_normalized = normalize(reference)
    difference = np.abs(native_normalized - reference_normalized)
    reference_l1 = float(
        np.abs(reference_normalized, dtype=np.float64).sum()
    )
    native_u8 = np.clip(native_normalized * 255.0, 0, 255).astype(np.uint8)
    reference_u8 = np.clip(
        reference_normalized * 255.0, 0, 255
    ).astype(np.uint8)
    u8_difference = np.abs(
        native_u8.astype(np.int16) - reference_u8.astype(np.int16)
    )
    return {
        "normalized_max_abs": float(difference.max()),
        "normalized_mean_abs": float(difference.mean()),
        "normalized_relative_l1": (
            float(difference.astype(np.float64).sum() / reference_l1)
            if reference_l1
            else 0.0
        ),
        "product_u8_max_abs": int(u8_difference.max()),
        "product_u8_mean_abs": float(u8_difference.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=CONFIGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--reference-device",
        choices=("cpu", "vulkan"),
        default="cpu",
    )
    parser.add_argument(
        "--inferbridge-contract",
        action="store_true",
        help="validate the deployed InferBridge preprocessing/output contract",
    )
    parser.add_argument(
        "--native-image-contract",
        action="store_true",
        help="exercise the native BGRA8-to-uint8 InferBridge image ABI",
    )
    parser.add_argument(
        "--reference-threads",
        type=int,
        default=0,
        help="override PyTorch CPU threads (zero keeps its default)",
    )
    parser.add_argument("--vulkan-device-index", type=int, default=0)
    args = parser.parse_args()
    if args.native_image_contract and not args.inferbridge_contract:
        parser.error("--native-image-contract requires --inferbridge-contract")
    if args.reference_threads > 0:
        torch.set_num_threads(args.reference_threads)

    config = CONFIGS[args.encoder]
    start = time.perf_counter()
    model = DepthAnything(config["model"])
    state = load_reference_state(args.checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or missing not in ([], ["pretrained.mask_token"]):
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.eval().to(args.reference_device)
    python_create_seconds = time.perf_counter() - start

    dll = configure_dll(args.dll)
    options = CreateOptions(
        ctypes.sizeof(CreateOptions),
        1,
        config["enum"],
        args.vulkan_device_index,
        0,
    )
    context = ctypes.c_void_p()
    start = time.perf_counter()
    status = dll.dav1_create(
        str(args.model.resolve()).encode(),
        ctypes.byref(options),
        ctypes.byref(context),
    )
    native_create_seconds = time.perf_counter() - start
    if status != 0:
        raise RuntimeError(dll.dav1_last_error().decode())

    paths = sorted(
        path
        for path in args.assets.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    preview_directory = args.output / args.encoder
    preview_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot decode {path}")
            height, width = image.shape[:2]
            tensor = (
                inferbridge_input(
                    image, args.input_size, args.reference_device
                )
                if args.inferbridge_contract
                else python_input(
                    image, args.input_size, args.reference_device
                )
            )
            native_is_u8 = args.native_image_contract
            if args.native_image_contract:
                shape = ImageShape()
                status = dll.dav1_get_inferbridge_shape(
                    width,
                    height,
                    args.input_size,
                    ctypes.byref(shape),
                )
                if status != 0:
                    raise RuntimeError(
                        f"{path}: {dll.dav1_last_error().decode()}"
                    )
                output_width, output_height = shape.width, shape.height
                if (output_height, output_width) != tuple(tensor.shape[-2:]):
                    raise RuntimeError(
                        f"{path}: native/Python output shape mismatch"
                    )
                native = np.empty(
                    output_width * output_height, dtype=np.uint8
                )
                bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                start = time.perf_counter()
                status = dll.dav1_inferbridge_bgra8_u8(
                    context,
                    bgra.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_uint8)
                    ),
                    width,
                    height,
                    bgra.strides[0],
                    args.input_size,
                    native.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_uint8)
                    ),
                    native.size,
                )
            elif args.inferbridge_contract:
                output_height, output_width = tensor.shape[-2:]
                tensor_cpu = np.ascontiguousarray(
                    tensor.cpu().numpy()[0], dtype=np.float32
                )
                native = np.empty(
                    output_width * output_height, dtype=np.float32
                )
                start = time.perf_counter()
                status = dll.dav1_infer_tensor_f32(
                    context,
                    tensor_cpu.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_float)
                    ),
                    output_width,
                    output_height,
                    native.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    native.size,
                )
            else:
                output_width, output_height = width, height
                native = np.empty(width * height, dtype=np.float32)
                start = time.perf_counter()
                status = dll.dav1_infer_bgr8(
                    context,
                    image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    width,
                    height,
                    image.strides[0],
                    args.input_size,
                    native.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    native.size,
                )
            native_seconds = time.perf_counter() - start
            if status != 0:
                raise RuntimeError(
                    f"{path}: {dll.dav1_last_error().decode()}"
                )

            start = time.perf_counter()
            with torch.inference_mode():
                reference = model(tensor)
                if args.inferbridge_contract:
                    reference = reference[0]
                else:
                    reference = torch.nn.functional.interpolate(
                        reference[:, None],
                        (height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
            reference = reference.cpu().numpy().astype(np.float32, copy=False)
            python_seconds = time.perf_counter() - start

            native_depth = native.reshape(output_height, output_width)
            if native_is_u8:
                reference_minimum = float(reference.min())
                reference_span = (
                    float(reference.max()) - reference_minimum
                )
                reference_product = (
                    np.zeros_like(reference, dtype=np.uint8)
                    if reference_span == 0.0
                    else np.clip(
                        (reference - reference_minimum) /
                        reference_span * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                )
                product_difference = np.abs(
                    native_depth.astype(np.int16) -
                    reference_product.astype(np.int16)
                )
                difference = (
                    product_difference.astype(np.float32) / 255.0
                )
                reference_for_l1 = (
                    reference_product.astype(np.float32) / 255.0
                )
            else:
                difference = np.abs(native_depth - reference)
                reference_for_l1 = reference
            differing = int(np.count_nonzero(
                native_depth != (
                    reference_product if native_is_u8 else reference
                )
            ))
            reference_l1 = float(
                np.abs(reference_for_l1, dtype=np.float64).sum()
            )
            row = {
                "encoder": args.encoder,
                "reference_device": args.reference_device,
                "vulkan_device_index": args.vulkan_device_index,
                "image": path.as_posix(),
                "width": output_width,
                "height": output_height,
                "pixels": output_width * output_height,
                "differing": differing,
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "rmse": float(
                    np.sqrt(
                        np.square(difference, dtype=np.float64).mean()
                    )
                ),
                "relative_l1": (
                    float(
                        difference.astype(np.float64).sum() /
                        reference_l1
                    )
                    if reference_l1
                    else 0.0
                ),
                "reference_min": float(reference_for_l1.min()),
                "reference_max": float(reference_for_l1.max()),
                "native_seconds": native_seconds,
                "python_seconds": python_seconds,
            }
            if native_is_u8:
                row.update({
                    "normalized_max_abs": float(difference.max()),
                    "normalized_mean_abs": float(difference.mean()),
                    "normalized_relative_l1": (
                        float(
                            difference.astype(np.float64).sum() /
                            reference_l1
                        )
                        if reference_l1
                        else 0.0
                    ),
                    "product_u8_max_abs": int(product_difference.max()),
                    "product_u8_mean_abs": float(
                        product_difference.mean()
                    ),
                })
            else:
                row.update(normalized_metrics(native_depth, reference))
            rows.append(row)

            depth = native_depth
            if native_is_u8:
                preview = depth
            else:
                span = float(depth.max() - depth.min())
                preview = (
                    np.zeros_like(depth, dtype=np.uint8)
                    if span == 0.0
                    else np.clip(
                        (depth - depth.min()) * (255.0 / span), 0, 255
                    ).astype(np.uint8)
                )
            relative = path.relative_to(args.assets)
            preview_name = "__".join(relative.parts)
            cv2.imwrite(str(preview_directory / f"{preview_name}.png"), preview)
            print(
                f"[{index:02d}/{len(paths)}] {path}: "
                f"diff={differing}, max={difference.max():.9g}, "
                f"C={native_seconds:.3f}s, Python={python_seconds:.3f}s",
                flush=True,
            )
    finally:
        dll.dav1_destroy(context)

    csv_path = (
        args.output /
        f"{args.encoder}_{args.reference_device}.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{args.encoder}: create C={native_create_seconds:.3f}s, "
        f"Python={python_create_seconds:.3f}s; results={csv_path}"
    )


if __name__ == "__main__":
    main()
