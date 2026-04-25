from __future__ import annotations

from typing import Any


def resolve_torch_dtype(dtype_name: str | None, torch_module: Any) -> Any:
    if dtype_name is None:
        return None
    normalized = dtype_name.strip().lower()
    if normalized in {"", "default", "float32", "fp32", "full", "no", "none"}:
        return None
    if normalized == "auto":
        return "auto"
    if normalized in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch_module.float16
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def model_load_kwargs(model_config: Any, torch_module: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    dtype = resolve_torch_dtype(getattr(model_config, "torch_dtype", None), torch_module)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if getattr(model_config, "low_cpu_mem_usage", False):
        kwargs["low_cpu_mem_usage"] = True
    attn_implementation = getattr(model_config, "attn_implementation", None)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def autocast_dtype(mixed_precision: str | None, torch_module: Any) -> Any:
    normalized = (mixed_precision or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    return None
