"""TraceContext：请求级追踪上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class TraceStage:
    """单个阶段记录。"""

    stage: str
    elapsed_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class TraceContext:
    """请求级追踪上下文。

    设计说明：
    1. `trace_type` 用于区分 query / ingestion 两条链路。
    2. `record_stage()` 只负责记录阶段快照，不修改全局结束态。
    3. `finish()` 必须幂等，便于多处安全调用。
    """

    def __init__(self, trace_type: str = "query") -> None:
        self.trace_id = str(uuid4())
        self.trace_type = trace_type
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.finished_at: Optional[str] = None
        self._started = perf_counter()
        self._finished_perf: Optional[float] = None
        self._stages: List[TraceStage] = []

    def record_stage(self, stage: str, elapsed_ms: float, **details: Any) -> None:
        """记录单个阶段结果。"""
        self._stages.append(
            TraceStage(stage=stage, elapsed_ms=float(elapsed_ms), details=dict(details))
        )

    def finish(self) -> None:
        """标记 trace 结束并冻结总耗时。"""
        if self._finished_perf is not None:
            return
        self._finished_perf = perf_counter()
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """获取指定阶段或整个 trace 的耗时。"""
        if stage_name is None:
            end = self._finished_perf if self._finished_perf is not None else perf_counter()
            return round((end - self._started) * 1000, 3)

        for item in self._stages:
            if item.stage == stage_name:
                return float(item.elapsed_ms)
        raise KeyError(f"stage not found: {stage_name}")

    def to_dict(self) -> Dict[str, Any]:
        """输出可直接 JSON 序列化的结构化字典。"""
        self.finish()
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": [self._stage_to_dict(stage) for stage in self._stages],
        }

    def get_stage(self, stage: str) -> Optional[Dict[str, Any]]:
        """按阶段名获取首条记录（测试辅助方法）。"""
        for item in self._stages:
            if item.stage == stage:
                return self._stage_to_dict(item)
        return None

    @staticmethod
    def _stage_to_dict(stage: TraceStage) -> Dict[str, Any]:
        payload = {
            "stage": stage.stage,
            "elapsed_ms": stage.elapsed_ms,
            "recorded_at": stage.recorded_at,
        }
        # 同时保留扁平字段与 details，兼顾调试可读性和向后兼容。
        payload.update(stage.details)
        payload["details"] = dict(stage.details)
        return payload
