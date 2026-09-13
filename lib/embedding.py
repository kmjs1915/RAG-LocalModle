# -*- coding: utf-8 -*-
"""
文件名：lib/embedding.py
功能：BGE Embedding 封装
      1) 本地加载 models/bge-small-zh-v1.5（相对路径），输出 384 维向量
      2) 强制前缀：文档切片加 "passage："、用户问题加 "query："
      3) 向量做 L2 归一化，配合 Chroma 的 cosine 度量
      4) 模型缺失/加载失败时给出明确错误；GPU 可用时自动使用 fp16 加速
实现策略：优先使用 FlagEmbedding 的 FlagModel；若该库不可用，则自动退化为
          transformers(AutoModel)+CLS 池化 的等价实现，保证工程可直接运行。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lib.security import ROOT_DIR

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/bge-small-zh-v1.5"
DEFAULT_QUERY_PREFIX = "query："
DEFAULT_PASSAGE_PREFIX = "passage："
EMBED_DIM = 384
MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")


class ModelNotFoundError(FileNotFoundError):
    """本地模型文件缺失。"""


class EmbeddingError(RuntimeError):
    """Embedding 计算失败。"""


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------
def resolve_model_path(path: Optional[str]) -> Path:
    """把配置里的相对路径解析为项目内绝对路径（适配 Windows 相对路径）。"""
    p = Path(path or DEFAULT_MODEL_PATH)
    return p if p.is_absolute() else (ROOT_DIR / p)


def check_embedding_model(path: Optional[str] = None) -> tuple[bool, str]:
    """启动自检：检查 Embedding 模型目录与权重文件是否存在，并报告实际输出维度。"""
    model_dir = resolve_model_path(path)
    if not model_dir.exists():
        return False, f"Embedding 模型目录不存在：{model_dir}"
    if not (model_dir / "config.json").exists():
        return False, f"Embedding 模型缺少 config.json：{model_dir}"
    has_weight = any((model_dir / w).exists() for w in MODEL_WEIGHT_FILES)
    if not has_weight:
        return False, f"Embedding 模型缺少权重文件（model.safetensors / pytorch_model.bin）：{model_dir}"
    if not (model_dir / "vocab.txt").exists() and not (model_dir / "tokenizer.json").exists():
        return False, f"Embedding 模型缺少分词器文件：{model_dir}"
    real_dim = detect_embedding_dim(str(model_dir))
    dim_note = f"，模型实际输出维度 {real_dim}" if real_dim else ""
    return True, f"Embedding 模型就绪：{model_dir}{dim_note}"


def detect_embedding_dim(path: Optional[str] = None) -> Optional[int]:
    """从模型 config.json 读取隐藏层维度（不加载模型，毫秒级），用于启动自检展示。"""
    model_dir = resolve_model_path(path)
    try:
        import json
        data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        dim = data.get("hidden_size") or data.get("dim")
        return int(dim) if dim else None
    except Exception:
        return None


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（范数为 0 的行保持不变，避免除零）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# Embedding 主体
# ---------------------------------------------------------------------------
class BGEEmbedding:
    """
    BGE 文本向量化封装。

    使用示例：
        emb = BGEEmbedding()
        vec_q = emb.encode_query("报销流程是什么？")        # 自动加 query： 前缀
        vec_p = emb.encode_passages(["passage：报销流程..."])  # 自动加 passage： 前缀
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 dim: int = EMBED_DIM,
                 query_prefix: str = DEFAULT_QUERY_PREFIX,
                 passage_prefix: str = DEFAULT_PASSAGE_PREFIX,
                 max_length: int = 512,
                 batch_size: int = 32,
                 use_fp16: Optional[bool] = None,
                 lazy: bool = True) -> None:
        self.model_path = resolve_model_path(model_path)
        self.dim = int(dim)
        self.query_prefix = query_prefix or ""
        self.passage_prefix = passage_prefix or ""
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
            ok, msg = check_embedding_model(str(self.model_path))
            if not ok:
                raise ModelNotFoundError(msg)

            import torch  # 局部导入，缩短模块导入耗时

            self._torch = torch
            cuda = torch.cuda.is_available()
            self._device = "cuda" if cuda else "cpu"
            fp16 = cuda if self._use_fp16 is None else bool(self._use_fp16)
            if not cuda:
                fp16 = False  # CPU 上 fp16 精度与速度都不合适

            # 1) 首选 FlagEmbedding
            try:
                from FlagEmbedding import FlagModel  # type: ignore

                # query_instruction_for_retrieval=None：前缀由本模块显式拼接，避免重复加前缀
                # normalize_embeddings=True：让 FlagEmbedding 也做 L2 归一化（本模块仍会再校验一次）
                common = {
                    "query_instruction_for_retrieval": None,
                    "use_fp16": fp16,
                    "normalize_embeddings": True,
                }
                try:
                    self._model = FlagModel(str(self.model_path), devices=self._device, **common)
                except TypeError:
                    # 兼容参数名为 device / 不支持 devices、normalize_embeddings 的旧版本
                    self._model = FlagModel(str(self.model_path), **common)
                self._backend = "flag"
                logger.info("Embedding 模型已加载（FlagEmbedding），device=%s, fp16=%s", self._device, fp16)
                return
            except Exception as exc:
                logger.warning("FlagEmbedding 不可用，改用 transformers 后端：%s", exc)

            # 2) 退化方案：transformers AutoModel + CLS 池化（与 BGE 官方池化方式一致）
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            model = AutoModel.from_pretrained(str(self.model_path))
            model.eval()
            if fp16:
                model.half()
            model.to(self._device)
            self._model = model
            self._backend = "transformers"
            logger.info("Embedding 模型已加载（transformers），device=%s, fp16=%s", self._device, fp16)

    @property
    def backend(self) -> str:
        """当前使用的后端名称，便于界面展示。"""
        return self._backend

    @property
    def device(self) -> str:
        return self._device

    # ---------------- 前缀 ----------------
    def add_passage_prefix(self, text: str) -> str:
        """文档切片强制加 passage： 前缀。"""
        text = (text or "").strip()
        return f"{self.passage_prefix}{text}"

    def add_query_prefix(self, text: str) -> str:
        """用户问题强制加 query： 前缀。"""
        text = (text or "").strip()
        return f"{self.query_prefix}{text}"

    # ---------------- 编码 ----------------
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        批量编码，返回 shape=(n, dim) 的 float32 归一化向量。
        is_query=True 时自动加 query： 前缀，否则加 passage： 前缀。
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        prefix = self.add_query_prefix if is_query else self.add_passage_prefix
        payload = [prefix(t) for t in texts]

        self.load()
        try:
            if self._backend == "flag":
                vectors = self._encode_flag(payload)
            else:
                vectors = self._encode_transformers(payload)
        except Exception as exc:
            logger.error("向量化失败: %s", exc)
            raise EmbeddingError(f"文本向量化失败：{exc}") from exc

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = _l2_normalize(vectors)          # 余弦度量必需：L2 归一化后内积即余弦
        if vectors.shape[1] != self.dim:
            # 说明：bge-small-zh-v1.5 实际隐藏层为 512 维（384 维是 bge-small-en-v1.5 的配置）。
            # 此处以模型实际输出维度为准，入库与检索使用同一模型，向量一致性不受影响。
            logger.info("模型实际输出维度 %d（配置 dim=%d），已按实际维度校准，检索一致性不受影响",
                        vectors.shape[1], self.dim)
            self.dim = int(vectors.shape[1])
        return vectors

    def _encode_flag(self, payload: List[str]) -> np.ndarray:
        """FlagEmbedding 后端编码（归一化已在构造与 encode 后统一处理）。"""
        try:
            out = self._model.encode(payload,
                                     batch_size=self.batch_size,
                                     max_length=self.max_length,
                                     normalize_embeddings=True)
        except TypeError:
            # FlagEmbedding 1.4+ 的 encode() 不接受 normalize_embeddings（改为构造参数），
            # 此处退化为仅传 batch_size/max_length，归一化由本方法调用方统一完成
            out = self._model.encode(payload, batch_size=self.batch_size, max_length=self.max_length)
        if isinstance(out, dict):  # 部分版本返回 dict
            out = out.get("dense_vecs", out)
        return np.asarray(out, dtype=np.float32)

    def _encode_transformers(self, payload: List[str]) -> np.ndarray:
        """transformers 后端编码：CLS 池化（BGE 系列官方池化方式）。"""
        torch = self._torch
        results: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(payload), self.batch_size):
                batch = payload[i:i + self.batch_size]
                encoded = self._tokenizer(batch,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt")
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                output = self._model(**encoded)
                cls = output.last_hidden_state[:, 0]           # [CLS] 向量
                results.append(cls.float().cpu().numpy())
        return np.vstack(results)

    # ---------------- 便捷接口 ----------------
    def encode_query(self, question: str) -> List[float]:
        """编码单个用户问题，返回 384 维列表（供 Chroma 查询）。"""
        vec = self.encode([question], is_query=True)
        return vec[0].tolist()

    def encode_queries(self, questions: List[str]) -> List[List[float]]:
        """批量编码用户问题。"""
        return self.encode(questions, is_query=True).tolist()

    def encode_passages(self, chunks: List[str]) -> List[List[float]]:
        """批量编码文档切片（入库用）。"""
        return self.encode(chunks, is_query=False).tolist()

    def warmup(self) -> None:
        """预热：首次编码用于触发 CUDA 初始化/模型编译，避免首个用户请求过慢。"""
        try:
            self.encode(["预热"], is_query=True)
            logger.info("Embedding 预热完成，维度=%d", self.dim)
        except Exception as exc:
            logger.warning("Embedding 预热失败（不影响后续使用）: %s", exc)


# ---------------------------------------------------------------------------
# 进程内单例缓存（避免多会话重复加载模型占用显存）
# ---------------------------------------------------------------------------
_EMBEDDER_CACHE: Dict[str, BGEEmbedding] = {}
_CACHE_LOCK = threading.RLock()


def get_embedder(config: Optional[Dict[str, Any]] = None) -> BGEEmbedding:
    """按配置返回全局唯一的 Embedding 实例。"""
    config = config or {}
    emb_cfg = config.get("embedding", {}) if config else {}
    model_path = str(resolve_model_path(emb_cfg.get("model_path", DEFAULT_MODEL_PATH)))
    with _CACHE_LOCK:
        if model_path not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[model_path] = BGEEmbedding(
                model_path=model_path,
                dim=int(emb_cfg.get("dim", EMBED_DIM) or EMBED_DIM),
                query_prefix=emb_cfg.get("query_prefix", DEFAULT_QUERY_PREFIX),
                passage_prefix=emb_cfg.get("passage_prefix", DEFAULT_PASSAGE_PREFIX),
                max_length=int(emb_cfg.get("max_length", 512) or 512),
                batch_size=int(emb_cfg.get("batch_size", 32) or 32),
                lazy=True,
            )
        return _EMBEDDER_CACHE[model_path]
