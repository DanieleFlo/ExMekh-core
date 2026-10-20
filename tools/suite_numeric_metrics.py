"""Frozen numerical gate definitions for development oracles (NumPy only)."""

import numpy as np


def compare(reference, candidate):
    original_a, original_b = np.asarray(reference), np.asarray(candidate)
    a, b = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    if a.shape != b.shape or not a.size:
        raise ValueError("shape mismatch or empty tensors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite tensor")
    delta = a - b
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "relative_l2": float(
            np.linalg.norm(delta.ravel()) / max(float(np.linalg.norm(a.ravel())), 1e-12)
        ),
        "bit_exact": bool(
            original_a.dtype == original_b.dtype
            and original_a.tobytes() == original_b.tobytes()
        ),
    }


def embedding_metrics(reference, candidate):
    a, b = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    result = compare(reference, candidate)
    if a.ndim != 2 or a.shape[-1] not in (128, 256, 512, 768):
        raise ValueError("expected supported embedding matrix")
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if not an.all() or not bn.all():
        raise ValueError("zero embedding")
    result["cosine_min"] = float(np.min(np.sum(a * b, axis=-1) / (an * bn)))
    result["unit_norm_error_max"] = float(np.max(np.abs(bn - 1)))
    return result


def ssim_rgb(reference, candidate):
    a, b = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    compare(a, b)
    if a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 11:
        raise ValueError("SSIM requires an HWC RGB image, dimensions >=11")
    if np.array_equal(a, b):
        return 1.0
    weights = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5**2))
    weights /= weights.sum()

    def blur(image):
        for axis in (0, 1):
            pads = [(0, 0)] * 3
            pads[axis] = (5, 5)
            view = np.lib.stride_tricks.sliding_window_view(
                np.pad(image, pads, mode="reflect"), 11, axis=axis
            )
            image = np.tensordot(view, weights, axes=([-1], [0]))
        return image

    ma, mb = blur(a), blur(b)
    va, vb, covariance = (
        blur(a * a) - ma * ma,
        blur(b * b) - mb * mb,
        blur(a * b) - ma * mb,
    )
    c1, c2 = 0.01**2, 0.03**2
    return float(
        np.mean(
            ((2 * ma * mb + c1) * (2 * covariance + c2))
            / ((ma * ma + mb * mb + c1) * (va + vb + c2))
        )
    )


def rgba_metrics(reference, candidate):
    a, b = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    result = compare(reference, candidate)
    if a.ndim != 3 or a.shape[-1] != 4:
        raise ValueError("expected HWC RGBA")
    if a.min() < 0 or a.max() > 1 or b.min() < 0 or b.max() > 1:
        raise ValueError("RGBA outside [0,1]")
    result["ssim_rgb"] = ssim_rgb(a[..., :3], b[..., :3])
    result["alpha_mae"] = float(np.mean(np.abs(a[..., 3] - b[..., 3])))
    return result
