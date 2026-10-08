"""Full-vocabulary comparisons shared by Flash qualification tools."""

import numpy as np


def compare_teacher_logits(reference, candidate, targets):
    """Compare matched teacher rows; reject incomplete or non-finite dumps."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    targets = np.asarray(targets)
    if (reference.ndim != 2 or reference.shape != candidate.shape or
            not reference.size or targets.shape != (reference.shape[0],) or
            not np.issubdtype(targets.dtype, np.integer) or
            not np.isfinite(reference).all() or not np.isfinite(candidate).all() or
            np.any(targets < 0) or np.any(targets >= reference.shape[1])):
        raise ValueError("invalid or unmatched full-logit output")

    def log_probabilities(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    log_reference = log_probabilities(reference)
    log_candidate = log_probabilities(candidate)
    kl = np.maximum(0, (np.exp(log_reference) * (log_reference - log_candidate)).sum(axis=1))
    rows = np.arange(len(targets))
    loss_reference = -log_reference[rows, targets].mean()
    loss_candidate = -log_candidate[rows, targets].mean()
    norms = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    cosine = np.divide((reference * candidate).sum(axis=1), norms,
                       out=np.zeros_like(norms), where=norms != 0)
    # Two zero vectors have identical uniform distributions. Keep diagnostics
    # serializable and distinguish them from a single zero vector.
    cosine[(np.linalg.norm(reference, axis=1) == 0) &
           (np.linalg.norm(candidate, axis=1) == 0)] = 1
    return {
        "rows": len(targets), "vocab": reference.shape[1], "finite": True,
        "bit_exact": bool(np.array_equal(reference, candidate)),
        "max_abs": float(np.abs(reference - candidate).max()),
        "kl_max": float(kl.max()), "kl_mean": float(kl.mean()),
        "loss_reference": float(loss_reference), "loss_candidate": float(loss_candidate),
        "loss_delta": float(loss_candidate - loss_reference),
        "argmax_rows_equal": int((reference.argmax(axis=1) == candidate.argmax(axis=1)).sum()),
        "cosine_min": float(cosine.min()),
    }


def teacher_quality_passes(metrics):
    """Existing strict full-row screen, including loss and cosine."""
    return (metrics["finite"] and metrics["argmax_rows_equal"] == metrics["rows"] and
            metrics["kl_max"] <= .001 and metrics["cosine_min"] >= .999 and
            abs(metrics["loss_delta"]) <= .001)
