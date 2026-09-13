# -*- coding: utf-8 -*-
"""
文件名：lib/reranker.py
功能：BGE Cross-Encoder 重排模型封装
      1) 本地加载 models/bge-reranker-base（相对路径）
      2) 对 (用户问题, 召回片段) 两两打分，输出原始 logit
      3) logit 经 Sigmoid 归一化为 0~1 相关性分数
      4) 重排失败时不阻断问答：退化为使用向量相似度作为分数，并在日志中说明
实现策略：优先使用 FlagEmbedding 的 FlagReranker；若不可用，自动退化为
          transformers(AutoModelForSequenceClassification) 等价实现。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.score_eval import sigmoid
from lib.security import ROOT_DIR

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/bge-reranker-base"
MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")


class ModelNotFoundError(FileNotFoundError):
    """本地模型文件缺失。"""


class RerankerError(RuntimeError):
    """重排计算失败。"""


def resolve_model_path(path: Optional[str]) -> Path:
    """把配置里的相对路径解析为项目内绝对路径。"""
    p = Path(path or DEFAULT_MODEL_PATH)
    return p if p.is_absolute() else (ROOT_DIR / p)


def check_reranker_model(path: Optional[str] = None) -> Tuple[bool, str]:
    """启动自检：检查 Reranker 模型目录与权重文件是否存在。"""
    model_dir = resolve_model_path(path)
    if not model_dir.exists():
        return False, f"Reranker 模型目录不存在：{model_dir}"
    if not (model_dir / "config.json").exists():
        return False, f"Reranker 模型缺少 config.json：{model_dir}"
    if not any((model_dir / w).exists() for w in MODEL_WEIGHT_FILES):
        return False, f"Reranker 模型缺少权重文件：{model_dir}"
    if not (model_dir / "tokenizer.json").exists() and not (model_dir / "sentencepiece.bpe.model").exists():
        return False, f"Reranker 模型缺少分词器文件：{model_dir}"
    return True, f"Reranker 模型就绪：{model_dir}"


class BGEReranker:
    """
    BGE Cross-Encoder 重排封装。

    使用示例：
        rr = BGEReranker()
        scores = rr.score("报销流程是什么？", ["片段1", "片段2"])   # -> [0.93, 0.11]
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 max_length: int = 512,
                 batch_size: int = 16,
                 use_fp16: Optional[bool] = None,
                 lazy: bool = True) -> None:
        self.model_path = resolve_model_path(model_path)
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self._use_fp16 = use_fp16
        self._backend = "unloaded"          # flag | transformers
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._lock = threading.RLock()
        if not lazy:
            self.load()

    # ---------------- 加载 ----------------
    def load(self) -> None:
        """加载模型（线程安全，重复调用只加载一次）。"""
        with self._lock:
            if self._backend != "unloaded":
                return
            ok, msg = check_reranker_model(str(self.model_path))
            if not ok:
                raise ModelNotFoundError(msg)

            import torch

            self._torch = torch
            cuda = torch.cuda.is_available()
            self._device = "cuda" if cuda else "cpu"
            fp16 = cuda if self._use_fp16 is None else bool(self._use_fp16)
            if not cuda:
                fp16 = False

            try:
                from FlagEmbedding import FlagReranker  # type: ignore

                common = {"use_fp16": fp16}
                try:
                    self._model = FlagReranker(str(self.model_path), devices=self._device, **common)
                except TypeError:
                    self._model = FlagReranker(str(self.model_path), **common)
                self._backend = "flag"
                logger.info("Reranker 模型已加载（FlagEmbedding），device=%s, fp16=%s", self._device, fp16)
                return
            except Exception as exc:
                logger.warning("FlagEmbedding Reranker 不可用，改用 transformers 后端：%s", exc)

            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            model.eval()
            if fp16:
                model.half()
            model.to(self._device)
            self._model = model
            self._backend = "transformers"
            logger.info("Reranker 模型已加载（transformers），device=%s, fp16=%s", self._device, fp16)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def device(self) -> str:
        return self._device

    # ---------------- 打分 ----------------
    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        """
        对 (query, passage) 两两打分，返回 0~1 的 Sigmoid 分数列表（顺序与入参一致）。
        空片段直接给 0 分；单个片段异常不阻断整批打分。
        """
        passages = [p if isinstance(p, str) else str(p) for p in (passages or [])]
        if not passages:
            return []
        query = (query or "").strip()
        self.load()

        pairs = [(query, p) for p in passages]
        try:
            logits = self._raw_logits(pairs)
        except Exception as exc:
            logger.error("Reranker 打分失败: %s", exc)
            raise RerankerError(f"重排打分失败：{exc}") from exc

        scores: List[float] = []
        for text, logit in zip(passages, logits):
            if not text.strip():
                scores.append(0.0)
            else:
                scores.append(round(sigmoid(float(logit)), 6))
        return scores

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """直接对 (query, passage) 列表打分，返回 0~1 分数。"""
        if not pairs:
            return []
        self.load()
        try:
            logits = self._raw_logits(list(pairs))
        except Exception as exc:
            raise RerankerError(f"重排打分失败：{exc}") from exc
        return [round(sigmoid(float(v)), 6) for v in logits]

    def _raw_logits(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """获取 Cross-Encoder 原始 logit（未归一化）。"""
        if self._backend == "flag":
            return self._raw_logits_flag(pairs)
        return self._raw_logits_transformers(pairs)

    def _raw_logits_flag(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """FlagEmbedding 后端：compute_score(normalize=False) 返回原始 logit。"""
        try:
            out = self._model.compute_score(pairs,
                                            batch_size=self.batch_size,
                                            max_length=self.max_length,
                                            normalize=False)
        except TypeError:
            # 兼容参数较少的旧版本
            out = self._model.compute_score(pairs, normalize=False)
        if isinstance(out, (int, float)):
            return [float(out)]
        return [float(v) for v in out]

    def _raw_logits_transformers(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """transformers 后端：取分类头第一个 logit（bge-reranker 只有一个输出节点）。"""
        torch = self._torch
        logits: List[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                queries = [p[0] for p in batch]
                docs = [p[1] for p in batch]
                encoded = self._tokenizer(queries,
                                          docs,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt")
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                output = self._model(**encoded)
                head = output.logits
                # 兼容输出维度为 (batch, 1) 或 (batch, 2) 的情况
                values = head[:, 0] if head.dim() > 1 else head
                logits.extend([float(v) for v in values.float().cpu().tolist()])
        return logits

    def warmup(self) -> None:
        """预热：触发模型首次推理，降低首个请求延迟。"""
        try:
            self.score("预热", ["预热文本"])
            logger.info("Reranker 预热完成")
        except Exception as exc:
            logger.warning("Reranker 预热失败（不影响后续使用）: %s", exc)


# ---------------------------------------------------------------------------
# 进程内单例缓存
# ---------------------------------------------------------------------------
_CACHE: Dict[str, BGEReranker] = {}
_CACHE_LOCK = threading.RLock()


def get_reranker(config: Optional[Dict[str, Any]] = None) -> BGEReranker:
    """按配置返回全局唯一的 Reranker 实例。"""
    config = config or {}
    rr_cfg = config.get("reranker", {}) if config else {}
    model_path = str(resolve_model_path(rr_cfg.get("model_path", DEFAULT_MODEL_PATH)))
    with _CACHE_LOCK:
        if model_path not in _CACHE:
            _CACHE[model_path] = BGEReranker(
                model_path=model_path,
                max_length=int(rr_cfg.get("max_length", 512) or 512),
                batch_size=int(rr_cfg.get("batch_size", 16) or 16),
                lazy=True,
            )
        return _CACHE[model_path]
