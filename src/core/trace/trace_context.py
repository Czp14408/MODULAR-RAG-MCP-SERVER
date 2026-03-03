"""TraceContext 最小实现（C5 占位版）。"""

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
    """请求级追踪上下文（后续 F 阶段会继续增强）。"""

    def __init__(self, trace_type: str = "ingestion") -> None:
        self.trace_id = str(uuid4())
        self.trace_type = trace_type
        self._started = perf_counter()
        self._stages: List[TraceStage] = []

    def record_stage(self, stage: str, elapsed_ms: float, **details: Any) -> None:
        """记录单个阶段结果。"""
        self._stages.append(
            TraceStage(stage=stage, elapsed_ms=float(elapsed_ms), details=dict(details))
        )

    def finish(self) -> Dict[str, Any]:
        """返回最小汇总结构，便于测试断言与日志输出。"""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "elapsed_ms": round((perf_counter() - self._started) * 1000, 3),
            "stages": [self._stage_to_dict(stage) for stage in self._stages],
        }

    def to_dict(self) -> Dict[str, Any]:
        """语义别名：与 finish 一致。"""
        return self.finish()

    def get_stage(self, stage: str) -> Optional[Dict[str, Any]]:
        """按阶段名获取首条记录（测试辅助方法）。"""
        for item in self._stages:
            if item.stage == stage:
                return self._stage_to_dict(item)
        return None

    @staticmethod
    def _stage_to_dict(stage: TraceStage) -> Dict[str, Any]:
        return {
            "stage": stage.stage,
            "elapsed_ms": stage.elapsed_ms,
            "details": stage.details,
            "recorded_at": stage.recorded_at,
        }
