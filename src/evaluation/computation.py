"""
Computation profiling utilities for DR model evaluation.

Provides both theoretical and practical efficiency metrics:
- Parameter count and approximate model size
- MACs / FLOPs estimate from Conv2d and Linear layers
- On-device latency and throughput benchmark
- Accuracy/QWK normalized efficiency indicators
"""

from __future__ import annotations

import platform
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


def estimate_model_size_mb(model: nn.Module) -> float:
    """Approximate in-memory model size (parameters + buffers)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes
    return float(total_bytes / (1024 ** 2))


def estimate_macs(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: torch.device,
) -> int:
    """
    Estimate MACs using forward hooks on Conv2d and Linear layers.

    Note: This is an approximation and may undercount operations from unsupported ops.
    """
    model_was_training = model.training
    model.eval()

    total_macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor):
        nonlocal total_macs
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            return

        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_h = output.shape[2]
        out_w = output.shape[3]

        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups

        macs_per_output = (in_channels // groups) * kernel_h * kernel_w
        output_elements = batch_size * out_channels * out_h * out_w
        total_macs += int(output_elements * macs_per_output)

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor):
        nonlocal total_macs
        if not isinstance(output, torch.Tensor):
            return

        batch_size = output.shape[0] if output.ndim > 1 else 1
        total_macs += int(batch_size * module.in_features * module.out_features)

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    try:
        with torch.no_grad():
            dummy = torch.randn(*input_shape, device=device)
            _ = model(dummy)
    finally:
        for hook in hooks:
            hook.remove()
        if model_was_training:
            model.train()

    return int(total_macs)


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int],
    warmup_runs: int = 8,
    benchmark_runs: int = 30,
) -> Dict[str, float]:
    """Benchmark mean/STD latency and throughput for a fixed input shape."""
    model_was_training = model.training
    model.eval()

    use_cuda_sync = device.type == "cuda"
    x = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(max(1, warmup_runs)):
            _ = model(x)

        if use_cuda_sync:
            torch.cuda.synchronize(device)

        timings_ms = []
        for _ in range(max(1, benchmark_runs)):
            start = time.perf_counter()
            _ = model(x)
            if use_cuda_sync:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)

    if model_was_training:
        model.train()

    latency_ms_mean = float(np.mean(timings_ms))
    latency_ms_std = float(np.std(timings_ms))
    throughput = float((input_shape[0] * 1000.0) / max(latency_ms_mean, 1e-6))

    return {
        "latency_ms_mean": latency_ms_mean,
        "latency_ms_std": latency_ms_std,
        "throughput_images_per_sec": throughput,
    }


def get_device_info(device: torch.device) -> Dict[str, str]:
    """Collect device metadata for reproducible reporting."""
    info = {
        "device_type": device.type,
        "device_name": "CPU",
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "device_name": torch.cuda.get_device_name(idx),
                "cuda_compute_capability": f"{props.major}.{props.minor}",
                "vram_gb": f"{props.total_memory / (1024 ** 3):.2f}",
            }
        )

    return info


def compute_efficiency_scores(
    accuracy: float,
    qwk: float,
    total_params: int,
    flops: int,
) -> Dict[str, float]:
    """
    Compute model efficiency indicators that normalize performance by computation.

    score formula:
        score = 100 * (0.5 * accuracy + 0.5 * qwk) / (gflops * log(1 + params_m))
    """
    params_m = total_params / 1e6
    gflops = flops / 1e9

    accuracy_per_gflop = float(accuracy / max(gflops, 1e-9))
    qwk_per_gflop = float(qwk / max(gflops, 1e-9))

    denominator = max(gflops * float(np.log1p(params_m)), 1e-9)
    computation_efficiency_score = float(
        100.0 * (0.5 * accuracy + 0.5 * qwk) / denominator
    )

    return {
        "params_m": float(params_m),
        "gflops": float(gflops),
        "accuracy_per_gflop": accuracy_per_gflop,
        "qwk_per_gflop": qwk_per_gflop,
        "computation_efficiency_score": computation_efficiency_score,
    }


def profile_model_computation(
    model: nn.Module,
    device: torch.device,
    input_size: int = 224,
    profile_batch_size: int = 1,
    warmup_runs: int = 8,
    benchmark_runs: int = 30,
) -> Dict[str, object]:
    """Run full computation profiling for a model on the active device."""
    input_shape = (profile_batch_size, 3, input_size, input_size)

    params = count_parameters(model)
    macs = estimate_macs(model, input_shape=input_shape, device=device)
    flops = int(2 * macs)
    model_size_mb = estimate_model_size_mb(model)
    timing = benchmark_inference(
        model,
        device=device,
        input_shape=input_shape,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )

    return {
        "device": get_device_info(device),
        "input_shape": list(input_shape),
        "parameters_total": params["total"],
        "parameters_trainable": params["trainable"],
        "parameters_frozen": params["frozen"],
        "model_size_mb": model_size_mb,
        "macs": int(macs),
        "gmacs": float(macs / 1e9),
        "flops": int(flops),
        "gflops": float(flops / 1e9),
        **timing,
    }
