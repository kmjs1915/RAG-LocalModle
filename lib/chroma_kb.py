# -*- coding: utf-8 -*-
"""
文件名：lib/chroma_kb.py
功能：Chroma 向量库层（知识库门面）
      1) PersistentClient 本地持久化，集合 personal_kb 显式指定 cosine 距离度量
      2) 入库：文档切片向量化后写入（向量在外部计算，不用 Chroma 默认 embedding）
      3) 检索接口：问题向量化 → 余弦召回 Top10 → Reranker 重排 → Top3
      4) 清理/重初始化：删除并重建集合，写操作全程加文件锁，防止并发损坏数据库
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from lib.embedding import BGEEmbedding, get_embedder
from lib.file_loader import Chunk, prepare_chunks
from lib.reranker import BGEReranker, RerankerError, get_reranker
from lib.security import CHROMA_DIR, ROOT_DIR, get_rag_params, load_config

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "personal_kb"
LOCK_STALE_SECONDS = 1800          # 锁文件超过 30 分钟视为残留锁，可强制接管
LOCK_ACQUIRE_TIMEOUT = 900         # 等待锁最长 15 分钟（重初始化大库可能较久）
WRITE_BATCH_SIZE = 64              # 单次写入 Chroma 的切片数量


class KBError(RuntimeError):
    """知识库操作失败。"""


# ---------------------------------------------------------------------------
# 跨进程文件锁
# ---------------------------------------------------------------------------
_PROCESS_LOCK = threading.RLock()


@contextmanager
def chroma_write_lock(timeout: int = LOCK_ACQUIRE_TIMEOUT) -> Iterator[None]:
    """
    Chroma 写操作文件锁（跨线程 + 跨进程双重保护）。
    - 线程级：threading.RLock，带超时等待，避免同一进程内多会话互相无限等待
    - 进程级：O_CREAT|O_EXCL 创建锁文件，天然原子
    - 锁文件超过 LOCK_STALE_SECONDS 未更新视为残留锁，自动清理后重试
    - 获取失败（超时）抛 KBError，由上层转成「权限/占用」友好提示
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = CHROMA_DIR / ".chroma_write.lock"
    start = time.time()

    # 1) 线程级锁：同一进程内多会话串行化（同线程可重入）
    if not _PROCESS_LOCK.acquire(timeout=max(0.0, float(timeout))):
        raise KBError("知识库正在被其他会话操作，请稍后重试（等待线程锁超时）")

    fd: Optional[int] = None
    try:
        # 2) 进程级文件锁：防止多进程/多实例同时写入损坏数据库
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"pid={os.getpid()} time={time.strftime('%Y-%m-%d %H:%M:%S')}".encode("utf-8"))
                os.close(fd)
                fd = None
                break
            except FileExistsError:
                # 残留锁检测：锁文件长时间未更新则强制清理
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > LOCK_STALE_SECONDS:
                        logger.warning("检测到残留锁文件（%.0f 秒未更新），已强制清理", age)
                        lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                if time.time() - start > timeout:
                    raise KBError("知识库正在被其他会话操作，请稍后重试（获取文件锁超时）")
                time.sleep(0.5)
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            lock_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("释放锁文件失败: %s", exc)
        _PROCESS_LOCK.release()


def _resolve_dir(path: Optional[str]) -> Path:
    """解析持久化目录（配置为相对路径时相对于项目根目录）。"""
    p = Path(path or "chroma_db")
    return p if p.is_absolute() else (ROOT_DIR / p)


# ---------------------------------------------------------------------------
# 知识库主体
# ---------------------------------------------------------------------------
class ChromaKB:
    """
    Chroma 知识库门面：初始化 / 入库 / 删除 / 检索。

    使用示例：
        kb = ChromaKB()
        kb.reset()                              # 全量重建（管理员）
        result = kb.retrieve("报销标准是多少")   # 召回 + 重排
    """

    def __init__(self,
                 persist_dir: Optional[str] = None,
                 collection_name: str = DEFAULT_COLLECTION,
                 embedder: Optional[BGEEmbedding] = None,
                 reranker: Optional[BGEReranker] = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()
        chroma_cfg = self.config.get("chroma", {})
        self.persist_dir = _resolve_dir(persist_dir or chroma_cfg.get("persist_dir", "chroma_db"))
        self.collection_name = collection_name or chroma_cfg.get("collection", DEFAULT_COLLECTION)
        self._embedder = embedder
        self._reranker = reranker
        self._client: Any = None
        self._collection: Any = None
        self._client_lock = threading.RLock()

    # ---------------- 基础属性 ----------------
    @property
    def embedder(self) -> BGEEmbedding:
        """懒加载 Embedding 模型（单例）。"""
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    @property
    def reranker(self) -> BGEReranker:
        """懒加载 Reranker 模型（单例）。"""
        if self._reranker is None:
            self._reranker = get_reranker(self.config)
        return self._reranker

    def fresh_config(self) -> Dict[str, Any]:
        """
        【修复点 19｜禁止模式 2】读取磁盘最新配置。

        本对象在启动时缓存了一份 config 快照，管理员后续修改 config.json
        （如调整 chunk_size / top_k / 阈值）后必须能立即生效，因此 RAG 参数
        一律通过本方法从磁盘重新读取，读不到时才退回启动快照。
        """
        try:
            return load_config(force=True)
        except Exception as exc:
            logger.warning("读取磁盘最新配置失败，退回启动快照：%s", exc)
            return self.config

    def _get_client(self) -> Any:
        """创建/复用 Chroma 持久化客户端。"""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import chromadb
                try:
                    from chromadb.config import Settings

                    settings = Settings(anonymized_telemetry=False, allow_reset=False)
                    self._client = chromadb.PersistentClient(path=str(self.persist_dir), settings=settings)
                except Exception:
                    self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                logger.info("Chroma 客户端已初始化: %s (collection=%s)", self.persist_dir, self.collection_name)
            except Exception as exc:
                raise KBError(f"Chroma 初始化失败：{exc}") from exc
            return self._client

    def _create_collection(self) -> Any:
        """
        创建集合，显式指定距离度量为 cosine。
        Chroma 1.x 支持 metadata={"hnsw:space": "cosine"}（等价于 configuration 写法）。
        """
        client = self._get_client()
        try:
            return client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},   # 余弦距离：distance = 1 - cosine_similarity
            )
        except Exception as exc:
            logger.error("集合创建失败: %s", exc)
            raise KBError(f"集合「{self.collection_name}」创建失败：{exc}") from exc

    @property
    def collection(self) -> Any:
        """获取（必要时创建）集合对象。"""
        with self._client_lock:
            if self._collection is None:
                self._collection = self._create_collection()
            return self._collection

    def _refresh_collection(self) -> Any:
        """重建集合对象（删除或重初始化后调用）。"""
        with self._client_lock:
            self._collection = self._create_collection()
            return self._collection

    # ---------------- 状态查询 ----------------
    def count(self) -> int:
        """返回集合内切片总数；异常时返回 0。"""
        try:
            return int(self.collection.count())
        except Exception as exc:
            logger.error("统计切片数量失败: %s", exc)
            return 0

    def stats(self) -> Dict[str, Any]:
        """知识库状态：切片数、来源文件数、持久化目录。"""
        total = self.count()
        files: List[str] = []
        try:
            if total:
                data = self.collection.get(include=["metadatas"])
                files = sorted({str((m or {}).get("source", "unknown")) for m in (data.get("metadatas") or [])})
        except Exception as exc:
            logger.warning("获取来源文件列表失败: %s", exc)
        return {
            "collection": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "chunk_count": total,
            "file_count": len(files),
            "files": files,
        }

    # ---------------- 入库 ----------------
    def add_chunks(self,
                   chunks: Sequence[Chunk],
                   batch_size: int = WRITE_BATCH_SIZE,
                   progress: Optional[Callable[[float, str], None]] = None) -> int:
        """
        批量写入切片：向量在本地计算（带 passage： 前缀），原文与元数据一并写入。
        返回成功写入的切片数。
        """
        if not chunks:
            return 0
        col = self.collection
        total = len(chunks)
        written = 0
        for start in range(0, total, batch_size):
            batch = list(chunks[start:start + batch_size])
            texts = [c.text for c in batch]
            try:
                # 向量化：encode_passages 内部自动加 passage： 前缀并做 L2 归一化
                vectors = self.embedder.encode_passages(texts)
            except Exception as exc:
                logger.error("切片向量化失败（批次 %d）: %s", start, exc)
                raise KBError(f"切片向量化失败：{exc}") from exc

            ids: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            for c in batch:
                digest = hashlib.md5(f"{c.source}|{c.chunk_index}|{c.text}".encode("utf-8")).hexdigest()[:12]
                ids.append(f"{c.source}::{c.chunk_index}::{digest}::{uuid.uuid4().hex[:6]}")
                metadatas.append({
                    "source": c.source,            # 规范要求的最小元数据：来源文件名
                    "chunk_index": int(c.chunk_index),
                    "chars": len(c.text),
                })
            try:
                col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
                written += len(batch)
            except Exception as exc:
                logger.error("写入 Chroma 失败（批次 %d）: %s", start, exc)
                raise KBError(f"写入向量库失败：{exc}") from exc

            if progress:
                progress(written / total, f"已入库 {written}/{total} 条切片")
        logger.info("入库完成，共写入 %d 条切片", written)
        return written

    def reset(self,
              progress: Optional[Callable[[float, str], None]] = None,
              docs_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        重新初始化知识库（管理员功能 7）：
          1) 加文件锁
          2) 删除并重建 personal_kb 集合（cosine）
          3) 重新扫描 user_docs 全量切片入库
        返回执行结果字典。
        """
        rag = get_rag_params(self.fresh_config())      # 切片参数取磁盘最新配置
        with chroma_write_lock():
            if progress:
                progress(0.02, "已获取文件锁，正在重建集合…")
            self.drop_collection()
            self._refresh_collection()
            if progress:
                progress(0.08, "正在扫描并解析 user_docs …")
            report = prepare_chunks(chunk_size=rag["chunk_size"],
                                    chunk_overlap=rag["chunk_overlap"],
                                    docs_dir=docs_dir)

            def _sub_progress(ratio: float, msg: str) -> None:
                if progress:
                    # 解析占 10%，入库占 90%
                    progress(0.10 + 0.88 * max(0.0, min(1.0, ratio)), msg)

            written = 0
            if report.chunks:
                written = self.add_chunks(report.chunks, progress=_sub_progress)
            if progress:
                progress(1.0, "初始化完成")
        return {
            "ok": True,
            "chunk_count": written,
            "files_total": report.files_total,
            "files_ok": report.files_ok,
            "errors": report.errors,
            "summary": report.summary,
        }

    def drop_collection(self) -> None:
        """
        仅删除向量集合（管理员功能 8），保留 user_docs 原始文件。
        注意：本方法不含锁，调用方需自行加锁。
        """
        with self._client_lock:
            try:
                self._get_client().delete_collection(name=self.collection_name)
                logger.info("集合已删除: %s", self.collection_name)
            except Exception as exc:
                # 集合本就不存在时视为成功（首次初始化会走到这里）
                if "does not exist" in str(exc).lower() or "not found" in str(exc).lower():
                    logger.info("集合 %s 尚不存在，无需删除", self.collection_name)
                else:
                    logger.warning("删除集合时出现提示（可忽略）: %s", exc)
            self._collection = None

    def clear(self) -> Dict[str, Any]:
        """加锁清理知识库（只删向量集合，不动原始文档）。"""
        with chroma_write_lock():
            self.drop_collection()
            self._refresh_collection()
        return {"ok": True, "chunk_count": 0, "message": "知识库向量集合已清空，user_docs 原始文件保持不变"}

    def delete_by_source(self, source: str) -> int:
        """按来源文件名删除切片（可选维护接口）。"""
        with chroma_write_lock():
            try:
                data = self.collection.get(where={"source": source}, include=[])
                ids = data.get("ids") or []
                if ids:
                    self.collection.delete(ids=ids)
                logger.info("已删除来源 %s 的 %d 条切片", source, len(ids))
                return len(ids)
            except Exception as exc:
                raise KBError(f"按来源删除失败：{exc}") from exc

    # ---------------- 检索 ----------------
    def recall(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        向量召回：问题（加 query： 前缀）→ 384 维向量 → 余弦相似度 Top-K。
        返回候选列表，含原文、来源、距离与余弦相似度。
        """
        question = (question or "").strip()
        if not question:
            return []
        total = self.count()
        if total <= 0:
            return []                       # 空库直接返回，由上层走空召回兜底
        query_vector = self.embedder.encode_query(question)
        n_results = max(1, min(int(top_k), total))
        try:
            result = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("向量检索失败: %s", exc)
            raise KBError(f"向量检索失败：{exc}") from exc

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        candidates: List[Dict[str, Any]] = []
        for i, text in enumerate(docs):
            distance = float(dists[i]) if i < len(dists) and dists[i] is not None else 1.0
            meta = metas[i] if i < len(metas) and metas[i] else {}
            candidates.append({
                "id": ids[i] if i < len(ids) else "",
                "text": text or "",
                "source": str((meta or {}).get("source", "unknown")),
                "chunk_index": int((meta or {}).get("chunk_index", i)),
                "distance": round(distance, 6),
                # cosine 空间下 distance = 1 - 余弦相似度
                "similarity": round(1.0 - distance, 6),
                "recall_rank": i + 1,
            })
        return candidates

    def retrieve(self,
                 question: str,
                 top_k_recall: Optional[int] = None,
                 top_k_rerank: Optional[int] = None) -> Dict[str, Any]:
        """
        完整检索接口（规范 3.2 的 2~7 步）：
          1) 问题向量化（加 query： 前缀）
          2) Chroma 余弦召回 Top10
          3) Reranker 对 (问题, 片段) 打分（Sigmoid 0~1）
          4) 按分数降序重排，截取 Top3
        返回 dict：recalled(含 rerank 分数) / used / empty / rerank_ok
        """
        rag = get_rag_params(self.fresh_config())      # Top-K 等检索参数取磁盘最新配置
        k_recall = int(top_k_recall or rag["top_k_recall"])
        k_rerank = int(top_k_rerank or rag["top_k_rerank"])

        candidates = self.recall(question, top_k=k_recall)
        if not candidates:
            # 空召回兜底：交由上层终止流程
            return {"recalled": [], "used": [], "empty": True, "rerank_ok": True, "reason": "知识库未检索到相关资料"}

        rerank_ok = True
        try:
            scores = self.reranker.score(question, [c["text"] for c in candidates])
        except RerankerError as exc:
            # 重排失败降级：使用余弦相似度近似排序，保证问答链路不中断
            logger.error("重排失败，降级为向量相似度排序: %s", exc)
            scores = [c["similarity"] for c in candidates]
            rerank_ok = False
        except Exception as exc:
            logger.error("重排异常，降级为向量相似度排序: %s", exc)
            scores = [c["similarity"] for c in candidates]
            rerank_ok = False

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = round(float(score), 6)     # 0~1 Sigmoid 分数

        # 按重排分数降序（分数相同时保留召回顺序，排序稳定）
        recalled = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        for i, cand in enumerate(recalled):
            cand["rerank_rank"] = i + 1
        used = recalled[:k_rerank]

        return {
            "recalled": recalled,
            "used": used,
            "empty": False,
            "rerank_ok": rerank_ok,
            "reason": "",
        }


# ---------------------------------------------------------------------------
# 进程内单例
# ---------------------------------------------------------------------------
_KB_CACHE: Dict[str, ChromaKB] = {}
_KB_LOCK = threading.RLock()


def get_kb(config: Optional[Dict[str, Any]] = None, collection_name: Optional[str] = None) -> ChromaKB:
    """返回全局唯一的 ChromaKB 实例。"""
    config = config or load_config()
    chroma_cfg = config.get("chroma", {})
    persist = str(_resolve_dir(chroma_cfg.get("persist_dir", "chroma_db")))
    name = collection_name or chroma_cfg.get("collection", DEFAULT_COLLECTION)
    key = f"{persist}::{name}"
    with _KB_LOCK:
        if key not in _KB_CACHE:
            _KB_CACHE[key] = ChromaKB(persist_dir=persist, collection_name=name, config=config)
        return _KB_CACHE[key]
