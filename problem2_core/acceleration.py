"""保持问题二 M1 固定点不变的初值与外层迭代加速工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import apply_ice_no_slip, apply_water_free_slip, ice_active_face_masks
from .config import Problem2Config
from .state import CoupledState


@dataclass(frozen=True)
class CouplingInitialGuess:
    ice_u: np.ndarray
    ice_v: np.ndarray
    ocean_u: np.ndarray
    ocean_v: np.ndarray
    sea_surface: np.ndarray
    strategy: str
    fallback_reason: str = ""


@dataclass(frozen=True)
class AndersonDecision:
    vector: np.ndarray
    method: str
    fallback_reason: str = ""
    history_cleared: bool = False
    history_depth: int = 0


class ProtectedAnderson:
    """小内存 Anderson-II；任何保护触发均回退调用方给出的原 Aitken 步。"""

    def __init__(
        self,
        *,
        depth: int = 4,
        damping: float = 1.0,
        residual_rise_factor: float = 1.25,
        condition_limit: float = 1.0e10,
        step_ratio_limit: float = 5.0,
    ) -> None:
        if not 1 <= depth <= 5:
            raise ValueError("Anderson 深度必须位于 1--5")
        if not 0.0 < damping <= 1.0:
            raise ValueError("Anderson 阻尼必须位于 (0, 1]")
        if residual_rise_factor <= 1.0:
            raise ValueError("残差上升保护因子必须大于 1")
        if condition_limit <= 1.0 or step_ratio_limit <= 1.0:
            raise ValueError("Anderson 病态与步长保护阈值必须大于 1")
        self.depth = int(depth)
        self.damping = float(damping)
        self.residual_rise_factor = float(residual_rise_factor)
        self.condition_limit = float(condition_limit)
        self.step_ratio_limit = float(step_ratio_limit)
        self._x_history: list[np.ndarray] = []
        self._f_history: list[np.ndarray] = []
        self._last_residual: float | None = None
        self._mask_signature: np.ndarray | None = None
        self.accepted_steps = 0
        self.fallbacks = 0
        self.history_clears = 0
        self.clear_reasons: list[str] = []

    @property
    def history_depth(self) -> int:
        return max(len(self._f_history) - 1, 0)

    def clear(self, reason: str) -> None:
        self._x_history.clear()
        self._f_history.clear()
        self._last_residual = None
        self._mask_signature = None
        self.history_clears += 1
        self.clear_reasons.append(reason)

    def _seed(
        self,
        current: np.ndarray,
        residual_vector: np.ndarray,
        residual: float,
        mask_signature: np.ndarray,
    ) -> None:
        self._x_history.append(np.array(current, copy=True))
        self._f_history.append(np.array(residual_vector, copy=True))
        self._last_residual = float(residual)
        self._mask_signature = np.array(mask_signature, dtype=bool, copy=True)

    def _fallback(
        self,
        fallback: np.ndarray,
        *,
        reason: str,
        cleared: bool = False,
    ) -> AndersonDecision:
        self.fallbacks += 1
        return AndersonDecision(
            vector=np.array(fallback, copy=True),
            method="aitken_fallback",
            fallback_reason=reason,
            history_cleared=cleared,
            history_depth=self.history_depth,
        )

    def propose(
        self,
        *,
        current: np.ndarray,
        raw: np.ndarray,
        residual: float,
        fallback: np.ndarray,
        mask_signature: np.ndarray,
    ) -> AndersonDecision:
        current = np.asarray(current, dtype=float)
        raw = np.asarray(raw, dtype=float)
        fallback = np.asarray(fallback, dtype=float)
        mask_signature = np.asarray(mask_signature, dtype=bool).ravel()
        residual_vector = raw - current
        if current.shape != raw.shape or current.shape != fallback.shape:
            raise ValueError("Anderson 当前值、原映射与回退值形状必须一致")

        finite = bool(
            np.isfinite(residual)
            and np.all(np.isfinite(current))
            and np.all(np.isfinite(raw))
            and np.all(np.isfinite(fallback))
        )
        if not finite:
            self.clear("nonfinite_value")
            return self._fallback(fallback, reason="nonfinite_value", cleared=True)

        if self._mask_signature is not None and not np.array_equal(
            mask_signature, self._mask_signature
        ):
            self.clear("active_ice_mask_changed")
            self._seed(current, residual_vector, residual, mask_signature)
            return self._fallback(
                fallback, reason="active_ice_mask_changed", cleared=True
            )
        if (
            self._last_residual is not None
            and residual > self.residual_rise_factor * self._last_residual
        ):
            self.clear("coupling_residual_increased")
            self._seed(current, residual_vector, residual, mask_signature)
            return self._fallback(
                fallback, reason="coupling_residual_increased", cleared=True
            )

        self._x_history.append(np.array(current, copy=True))
        self._f_history.append(np.array(residual_vector, copy=True))
        self._last_residual = float(residual)
        self._mask_signature = np.array(mask_signature, copy=True)
        keep = self.depth + 1
        if len(self._x_history) > keep:
            self._x_history = self._x_history[-keep:]
            self._f_history = self._f_history[-keep:]
        if len(self._f_history) < 2:
            return self._fallback(fallback, reason="insufficient_history")

        delta_x = np.column_stack(
            [
                self._x_history[index + 1] - self._x_history[index]
                for index in range(len(self._x_history) - 1)
            ]
        )
        delta_f = np.column_stack(
            [
                self._f_history[index + 1] - self._f_history[index]
                for index in range(len(self._f_history) - 1)
            ]
        )
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            delta_f, residual_vector, rcond=None
        )
        columns = delta_f.shape[1]
        condition = (
            float("inf")
            if singular_values.size == 0 or singular_values[-1] <= np.finfo(float).tiny
            else float(singular_values[0] / singular_values[-1])
        )
        if rank < columns or not np.isfinite(condition) or condition > self.condition_limit:
            self.clear("ill_conditioned_or_rank_deficient")
            self._seed(current, residual_vector, residual, mask_signature)
            return self._fallback(
                fallback,
                reason="ill_conditioned_or_rank_deficient",
                cleared=True,
            )

        candidate = (
            current
            + self.damping * residual_vector
            - (delta_x + self.damping * delta_f) @ coefficients
        )
        fallback_step = float(np.linalg.norm(fallback - current))
        candidate_step = float(np.linalg.norm(candidate - current))
        if not np.all(np.isfinite(candidate)):
            self.clear("nonfinite_anderson_candidate")
            self._seed(current, residual_vector, residual, mask_signature)
            return self._fallback(
                fallback, reason="nonfinite_anderson_candidate", cleared=True
            )
        if candidate_step > self.step_ratio_limit * max(
            fallback_step, np.finfo(float).tiny
        ):
            self.clear("anderson_step_guard")
            self._seed(current, residual_vector, residual, mask_signature)
            return self._fallback(fallback, reason="anderson_step_guard", cleared=True)

        self.accepted_steps += 1
        return AndersonDecision(
            vector=candidate,
            method="anderson",
            history_depth=self.history_depth,
        )


def _previous_state_guess(current: CoupledState, reason: str) -> CouplingInitialGuess:
    return CouplingInitialGuess(
        ice_u=np.array(current.ice_u, copy=True),
        ice_v=np.array(current.ice_v, copy=True),
        ocean_u=np.array(current.ocean_u, copy=True),
        ocean_v=np.array(current.ocean_v, copy=True),
        sea_surface=np.array(current.sea_surface, copy=True),
        strategy="previous_accepted_state",
        fallback_reason=reason,
    )


def build_temporal_initial_guess(
    config: Problem2Config,
    *,
    current: CoupledState,
    previous: CoupledState | None,
) -> CouplingInitialGuess:
    """由前两个接受态线性外推耦合初值；无效时严格回退当前接受态。"""

    if not config.temporal_predictor_enabled:
        return _previous_state_guess(current, "temporal_predictor_disabled")
    if previous is None:
        return _previous_state_guess(current, "insufficient_accepted_history")
    if not np.isclose(current.time_seconds - previous.time_seconds, config.dt):
        return _previous_state_guess(current, "nonconsecutive_accepted_history")

    grid = config.grid
    active_center_current, active_u, active_v = ice_active_face_masks(
        grid, current.thickness, h_min=config.h_min
    )
    active_center_previous, _, _ = ice_active_face_masks(
        grid, previous.thickness, h_min=config.h_min
    )
    if np.any(active_center_current != active_center_previous):
        return _previous_state_guess(current, "active_ice_mask_changed")

    predicted = {
        "ice_u": 2.0 * current.ice_u - previous.ice_u,
        "ice_v": 2.0 * current.ice_v - previous.ice_v,
        "ocean_u": 2.0 * current.ocean_u - previous.ocean_u,
        "ocean_v": 2.0 * current.ocean_v - previous.ocean_v,
        "sea_surface": 2.0 * current.sea_surface - previous.sea_surface,
    }
    if not all(np.all(np.isfinite(value)) for value in predicted.values()):
        return _previous_state_guess(current, "nonfinite_temporal_prediction")

    ice_u, ice_v = apply_ice_no_slip(grid, predicted["ice_u"], predicted["ice_v"])
    ice_u[~active_u] = 0.0
    ice_v[~active_v] = 0.0
    ocean_u, ocean_v = apply_water_free_slip(
        grid, predicted["ocean_u"], predicted["ocean_v"]
    )
    ice_uc, ice_vc = grid.faces_to_center(ice_u, ice_v)
    ocean_uc, ocean_vc = grid.faces_to_center(ocean_u, ocean_v)
    if max(
        float(np.max(np.hypot(ice_uc, ice_vc))),
        float(np.max(np.hypot(ocean_uc, ocean_vc))),
    ) > config.maximum_speed:
        return _previous_state_guess(current, "predicted_speed_exceeds_guard")

    return CouplingInitialGuess(
        ice_u=ice_u,
        ice_v=ice_v,
        ocean_u=ocean_u,
        ocean_v=ocean_v,
        sea_surface=np.array(predicted["sea_surface"], copy=True),
        strategy="two_state_linear_predictor",
    )
