from __future__ import annotations

import numpy as np


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def damped_pseudoinverse(jacobian: np.ndarray, damping: float) -> np.ndarray:
    """Damped least-squares pseudoinverse J^# = J^T (J J^T + lambda I)^(-1)."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    regularized = jacobian @ jacobian.T + damping * np.eye(jacobian.shape[0], dtype=np.float64)
    return jacobian.T @ np.linalg.inv(regularized)


def nullspace_projector(jacobian: np.ndarray, damping: float) -> np.ndarray:
    j_pinv = damped_pseudoinverse(jacobian, damping)
    return np.eye(jacobian.shape[1], dtype=np.float64) - j_pinv @ jacobian


def directional_manipulability(jacobian: np.ndarray, direction: np.ndarray, damping: float) -> float:
    """Directional manipulability

    For translational Jacobian J(q) and unit direction d, the damped effort metric
    for generating unit velocity along d is

        d^T (J J^T + lambda I)^(-1) d

    Its inverse square root is a direction-specific efficiency score

        mu_d = (d^T (J J^T + lambda I)^(-1) d)^(-1/2)

    Larger mu_d means less joint-speed norm is required to move along d.
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    direction = normalize(direction)
    metric = jacobian @ jacobian.T + damping * np.eye(jacobian.shape[0], dtype=np.float64)
    inv_metric = np.linalg.inv(metric)
    denom = float(direction @ inv_metric @ direction)
    return float(denom ** -0.5)


def directional_manipulability_gradient(q: np.ndarray, mu_fn, step: float = 1e-3) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    grad = np.zeros_like(q, dtype=np.float64)
    eye = np.eye(q.size, dtype=np.float64)
    for i in range(q.size):
        grad[i] = (mu_fn(q + step * eye[i]) - mu_fn(q - step * eye[i])) / (2.0 * step)
    return grad


def linear_map_tanh_to_range(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    action = np.clip(action, -1.0, 1.0)
    return low + 0.5 * (action + 1.0) * (high - low)


def linear_map_range_to_tanh(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    normalized = 2.0 * (values - low) / (high - low) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    return normalize(vec)


def wrap_angle_difference(delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi
