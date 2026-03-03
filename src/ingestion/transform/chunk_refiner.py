"""ChunkRefiner：规则去噪 + 可选 LLM 增强（C5）。"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Tuple

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory


class ChunkRefiner(BaseTransform):
    """对 Chunk 文本进行清洗与增强，且保证失败不阻塞主流程。"""

    def __init__(
        self,
        settings: Any,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
    ) -> None:
        self.settings = settings
        self.use_llm = self._read_use_llm(settings)
        self.prompt = self._load_prompt(prompt_path=prompt_path)
        self._llm: Optional[BaseLLM] = llm
        self._llm_init_error: Optional[str] = None
        self._last_llm_error: Optional[str] = None

        # 参数选择说明：
        # 1) 仅当 use_llm=True 且未注入 llm 时，才尝试工厂创建，避免无谓初始化开销。
        # 2) 初始化失败不抛致命异常，记录错误并在 transform 中自动降级为 rule-only。
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(settings)
            except Exception as exc:  # noqa: BLE001
                self._llm_init_error = str(exc)
                self._llm = None

    def transform(self, chunks: List[Chunk], trace: Optional[TraceContext] = None) -> List[Chunk]:
        started = perf_counter()
        refined: List[Chunk] = []
        llm_success = 0
        llm_fallback = 0

        for chunk in chunks:
            try:
                rule_text = self._rule_based_refine(chunk.text)
                output_text = rule_text
                refined_by = "rule"
                fallback_reason: Optional[str] = None

                if self.use_llm:
                    llm_text = self._llm_refine(rule_text, trace)
                    if isinstance(llm_text, str) and llm_text.strip():
                        output_text = llm_text.strip()
                        refined_by = "llm"
                        llm_success += 1
                    else:
                        llm_fallback += 1
                        fallback_reason = self._last_llm_error or "empty_llm_output"

                metadata = dict(chunk.metadata)
                metadata["refined_by"] = refined_by
                if fallback_reason:
                    metadata["refine_fallback_reason"] = fallback_reason

                refined.append(replace(chunk, text=output_text, metadata=metadata))
            except Exception as exc:  # noqa: BLE001
                # 单个 chunk 异常不影响全局：保留原文并标记失败原因。
                metadata = dict(chunk.metadata)
                metadata["refined_by"] = "rule"
                metadata["refine_fallback_reason"] = f"chunk_exception:{exc}"
                refined.append(replace(chunk, metadata=metadata))

        if trace is not None:
            trace.record_stage(
                "chunk_refiner",
                elapsed_ms=(perf_counter() - started) * 1000,
                method="rule+llm" if self.use_llm else "rule-only",
                chunk_count=len(chunks),
                llm_enabled=self.use_llm,
                llm_success=llm_success,
                llm_fallback=llm_fallback,
            )
        return refined

    def _rule_based_refine(self, text: str) -> str:
        """规则去噪：清理常见噪声并保留代码块结构。"""
        if not isinstance(text, str):
            return ""

        # 先切分代码块，避免对代码内部做空白折叠与行清理。
        segments = self._split_code_and_text(text)
        cleaned: List[str] = []
        for segment, is_code in segments:
            if is_code:
                cleaned.append(segment.strip())
            else:
                cleaned.append(self._clean_plain_text(segment))

        merged = "\n".join(part for part in cleaned if part.strip())
        merged = re.sub(r"\n{3,}", "\n\n", merged)
        return merged.strip()

    def _llm_refine(self, text: str, trace: Optional[TraceContext]) -> Optional[str]:
        """LLM 增强：失败返回 None，并记录错误原因供降级路径使用。"""
        self._last_llm_error = None
        if not self.use_llm:
            return None
        if self._llm is None:
            self._last_llm_error = self._llm_init_error or "llm_not_initialized"
            return None

        llm_started = perf_counter()
        try:
            prompt = self.prompt.format(text=text)
            result = self._llm.chat(
                [
                    Message(role="system", content="你是一个中文技术文档清洗助手。"),
                    Message(role="user", content=prompt),
                ]
            )
            if trace is not None:
                trace.record_stage(
                    "chunk_refiner_llm_call",
                    elapsed_ms=(perf_counter() - llm_started) * 1000,
                    provider=self._resolve_llm_provider(),
                )
            return result
        except Exception as exc:  # noqa: BLE001
            self._last_llm_error = str(exc)
            return None

    def _load_prompt(self, prompt_path: Optional[str]) -> str:
        # 参数选择说明：
        # 1) 外部显式传入 prompt_path 优先级最高，方便测试注入。
        # 2) 未传入时读取 settings.ingestion.chunk_refiner.prompt_path。
        candidate = prompt_path or self._read_prompt_path(self.settings)
        path = Path(candidate)
        default_prompt = (
            "请清理以下文档片段中的噪声（页眉页脚、多余空白、HTML注释等），"
            "保留关键事实与原有术语，不要编造内容。\n\n{text}"
        )
        if not path.exists():
            return default_prompt
        content = path.read_text(encoding="utf-8").strip()
        if "{text}" not in content:
            content = f"{content}\n\n{{text}}"
        return content

    @staticmethod
    def _split_code_and_text(text: str) -> List[Tuple[str, bool]]:
        """按 fenced code block 分段，返回 (segment, is_code) 列表。"""
        pattern = re.compile(r"(```[\s\S]*?```)")
        parts: List[Tuple[str, bool]] = []
        cursor = 0
        for match in pattern.finditer(text):
            plain = text[cursor : match.start()]
            if plain:
                parts.append((plain, False))
            parts.append((match.group(1), True))
            cursor = match.end()
        tail = text[cursor:]
        if tail:
            parts.append((tail, False))
        return parts if parts else [(text, False)]

    def _clean_plain_text(self, text: str) -> str:
        """清理非代码段文本。"""
        text = re.sub(r"<!--[\s\S]*?-->", "", text)  # HTML 注释
        lines = text.splitlines()
        kept: List[str] = []
        for raw in lines:
            line = raw.strip()
            if self._is_noise_line(line):
                continue
            # 将行内多空白压成 1 个空格，提升 embedding 稳定性。
            normalized = re.sub(r"[ \t]{2,}", " ", line)
            kept.append(normalized)
        merged = "\n".join(kept)
        merged = re.sub(r"\n{3,}", "\n\n", merged)
        return merged.strip()

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        if not line:
            return True
        patterns = [
            r"^第\s*\d+\s*页$",
            r"^Page\s+\d+(\s*/\s*\d+)?$",
            r"^版权所有.*$",
            r"^[-_*]{3,}$",
            r"^\[\d+/\d+\]$",
        ]
        return any(re.match(p, line, flags=re.IGNORECASE) for p in patterns)

    @staticmethod
    def _read_use_llm(settings: Any) -> bool:
        if hasattr(settings, "ingestion") and hasattr(settings.ingestion, "chunk_refiner"):
            return bool(getattr(settings.ingestion.chunk_refiner, "use_llm", False))
        if isinstance(settings, dict):
            ingestion = settings.get("ingestion")
            if isinstance(ingestion, dict):
                refiner = ingestion.get("chunk_refiner")
                if isinstance(refiner, dict):
                    return bool(refiner.get("use_llm", False))
        return False

    @staticmethod
    def _read_prompt_path(settings: Any) -> str:
        if hasattr(settings, "ingestion") and hasattr(settings.ingestion, "chunk_refiner"):
            value = getattr(settings.ingestion.chunk_refiner, "prompt_path", None)
            if isinstance(value, str) and value.strip():
                return value
        if isinstance(settings, dict):
            ingestion = settings.get("ingestion")
            if isinstance(ingestion, dict):
                refiner = ingestion.get("chunk_refiner")
                if isinstance(refiner, dict):
                    value = refiner.get("prompt_path")
                    if isinstance(value, str) and value.strip():
                        return value
        return "config/prompts/chunk_refinement.txt"

    def _resolve_llm_provider(self) -> str:
        if hasattr(self.settings, "llm") and hasattr(self.settings.llm, "provider"):
            return str(self.settings.llm.provider)
        if isinstance(self.settings, dict):
            llm = self.settings.get("llm")
            if isinstance(llm, dict):
                return str(llm.get("provider", "unknown"))
        return "unknown"


def load_noisy_chunk_fixtures(path: str) -> Any:
    """测试辅助：读取 noisy fixture。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
