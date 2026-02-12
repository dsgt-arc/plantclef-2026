from __future__ import annotations

import numpy as np


def f16_bytes_list_to_f32_matrix(buffers: list[bytes], num_classes: int) -> np.ndarray:
    t = len(buffers)
    out = np.empty((t, num_classes), dtype=np.float32)
    for i, b in enumerate(buffers):
        v = np.frombuffer(b, dtype=np.float16)
        if v.size != num_classes:
            raise ValueError(f"Expected {num_classes} values but got {v.size}")
        out[i] = v.astype(np.float32)
    return out


def f16_vector_to_bytes(vec: np.ndarray) -> bytes:
    if vec.dtype != np.float16:
        vec = vec.astype(np.float16, copy=False)
    return vec.tobytes(order="C")
