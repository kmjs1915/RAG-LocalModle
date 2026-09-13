# -*- coding: utf-8 -*-
"""
文件名：lib/score_eval.py
功能：置信度计算、兜底逻辑与回答流程编排
      1) Sigmoid 归一化（供 reranker 复用），数值稳定实现
      2) 置信度 = Top3 重排分数平均值 × 100，保留一位小数
      3) 两级兜底：空召回兜底、低置信度兜底
      4) 问题清洗与超长截断保护
      5) answer_question()：串联「清洗 → 召回 → 重排 → 置信度 → LLM → 追加置信度文本」
         （规范 3.2 的 1~11 步完整流程，供 app.py 直接调用）
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lib.security import get_rag_params, load_config

logger = logging.getLogger(__name__)

# 兜底与提示文案（规范原文）
NO_RESULT_TEXT = "知识库未检索到相关资料"
# 【需求2】全部片段余弦 < 0.5 时返回的低置信提示文案（main.py 直接引用本常量传给前端）
LOW_CONFIDENCE_TEXT = "知识库未找到高度匹配资料，回答仅供参考。"
EMPTY_QUESTION_TEXT = "请输入你的问题后再发送。"
LLM_FAIL_TIP = "（大模型回答生成失败，以下为检索到的资料摘要）"
CONFIDENCE_LABEL = "参考资料置信度"

# 【需求2】置信度过滤阈值：余弦相似度 < 0.5 的片段不参与平均值计算
COSINE_FILTER_THRESHOLD = 0.5

# 置信度阈值（与 config.json rag.low_confidence_threshold 保持一致，0.4 以下触发兜底）

# Sigmoid 数值稳定实现：x 过大/过小直接取极限，避免 exp 溢出
_SIGMOID_CLAMP = 60.0


def sigmoid(x: float) -> float:
    """
    Sigmoid 归一化：把 Reranker 的原始 logit 映射到 0~1。
    x > 0 时用 1/(1+e^-x)，x <= 0 时用 e^x/(1+e^x)，避免 exp 溢出。
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    x = max(-_SIGMOID_CLAMP, min(_SIGMOID_CLAMP, x))
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


# ---------------------------------------------------------------------------
# 问题清洗
# ---------------------------------------------------------------------------
def clean_question(question: str, max_chars: int = 2000) -> str:
    """
    清洗用户问题：
      1) 统一换行、压缩连续空格与多余空行
      2) 去掉首尾空白
      3) 超长截断保护（默认 2000 字符），避免超长文本导致模型报错
    """
    text = question if isinstance(question, str) else str(question or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()
    if max_chars and len(text) > max_chars:
        logger.warning("用户问题超长，已截断至 %d 字符", max_chars)
        text = text[:max_chars]
    return text


# ---------------------------------------------------------------------------
# 置信度计算
# ---------------------------------------------------------------------------
def average_score(scores: List[float]) -> float:
    """计算片段分数平均值（空列表返回 0.0）。"""
    values = [float(s) for s in (scores or []) if isinstance(s, (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def chunk_cosine(chunk: Dict[str, Any]) -> float:
    """
    取出单个片段的**原始余弦相似度**（需求2 的输入）。

    chroma_kb.recall() 写入的 similarity = 1 - cosine_distance，即原始余弦值（约 -1~1）；
    取值失败或缺字段时按 0.0 处理（自然会被 0.5 阈值过滤掉，不参与平均）。
    """
    try:
        value = chunk.get("similarity")
        if value is None:
            logger.warning("片段缺少 similarity 字段，按余弦 0.0 处理：来源=%s",
                           str(chunk.get("source", "?")))
            return 0.0
        return float(value)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def compute_confidence(chunks: List[Dict[str, Any]],
                       threshold: float = COSINE_FILTER_THRESHOLD) -> Dict[str, Any]:
    """
    置信度评估（需求2：原始余弦相似度 + 阈值过滤，不使用 Reranker 分数、不做归一化）：

      1) 入参为送入 LLM 的 Top-K（默认 Top3）片段，逐个取**原始余弦相似度**；
      2) 过滤：余弦 < threshold（默认 0.5）的片段**不参与**平均值计算；
      3) 分支判断：
         情况 A：至少 1 个片段余弦 ≥ 0.5 → 置信度 = 这些片段的算术平均值（× 100）；
         情况 B：全部片段余弦 < 0.5    → 置信度标记为 0，并返回低置信提示文案；
      4) 不做局部 min-max 归一化，直接使用原始余弦相似度。

    返回 dict：
      avg            参与计算片段的平均余弦（情况 B 为 0.0）
      confidence     0~100 的置信度（保留一位小数）
      text           展示文案「参考资料置信度：XX.X%」
      tips           需随回答一并展示的提示文案（情况 B 为低置信提示，否则空串）
      low_confidence 是否命中情况 B（前端据此展示提示文案）
      scores         全部片段的原始余弦列表
      used_scores    实际参与平均的余弦列表（≥ threshold）
      filtered_out   被过滤掉的片段数
      threshold      本次使用的过滤阈值
    """
    threshold = float(threshold)
    scores = [chunk_cosine(c) for c in (chunks or [])]
    used_scores = [s for s in scores if s >= threshold]       # 只保留余弦 ≥ 0.5 的片段

    if used_scores:
        # 情况 A：只用 ≥0.5 的片段求算术平均
        avg = average_score(used_scores)
        low_confidence = False
        tips = ""
    else:
        # 情况 B：3 个片段全部 <0.5 → 置信度标记为 0，并给出提示文案
        avg = 0.0
        low_confidence = True
        tips = LOW_CONFIDENCE_TEXT

    confidence = round(avg * 100, 1)
    text = f"{CONFIDENCE_LABEL}：{confidence:.1f}%"
    logger.info("[置信度] 原始余弦=%s 参与平均=%s 阈值=%.2f 置信度=%.1f%% 分支=%s",
                [round(s, 4) for s in scores], [round(s, 4) for s in used_scores],
                threshold, confidence, "B(全部低于阈值)" if low_confidence else "A")
    return {
        "avg": avg,
        "confidence": confidence,
        "text": text,
        "tips": tips,
        "low_confidence": low_confidence,
        "scores": scores,
        "used_scores": used_scores,
        "filtered_out": len(scores) - len(used_scores),
        "threshold": threshold,
    }


def format_confidence_text(avg: float) -> str:
    """按规范格式输出置信度文本：参考资料置信度：XX.X%"""
    return f"{CONFIDENCE_LABEL}：{round(float(avg) * 100, 1):.1f}%"


# ---------------------------------------------------------------------------
# 回答结果结构
# ---------------------------------------------------------------------------
@dataclass
class RagResult:
    """一次问答的完整结果，供 UI 渲染聊天区与调试面板。"""
    question: str = ""
    answer: str = ""                                   # 展示给用户的完整回答（含置信度文本）
    llm_answer: str = ""                               # 模型原始回答
    confidence: float = 0.0                            # 0~100（基于 Top3 原始余弦相似度）
    confidence_avg: float = 0.0                        # 参与平均的原始余弦均值（0~1）
    confidence_text: str = ""
    tips: str = ""                                     # 【需求2】随回答一并给前端的提示文案
    low_confidence: bool = False
    no_result: bool = False                            # 是否触发空召回兜底
    error: str = ""                                    # 友好错误提示（非空表示流程异常）
    recalled: List[Dict[str, Any]] = field(default_factory=list)   # 召回 Top10（含重排分数）
    used: List[Dict[str, Any]] = field(default_factory=list)       # 送入 LLM 的 Top3
    rerank_ok: bool = True
    provider: str = ""
    elapsed_ms: int = 0

    def to_debug_dict(self) -> Dict[str, Any]:
        """调试面板所需的精简结构。"""
        return {
            "question": self.question,
            "no_result": self.no_result,
            "low_confidence": self.low_confidence,
            "confidence": self.confidence,
            "rerank_ok": self.rerank_ok,
            "provider": self.provider,
            "elapsed_ms": self.elapsed_ms,
            "recalled": self.recalled,
            "used": self.used,
        }


# ---------------------------------------------------------------------------
# 回答流程编排（规范 3.2 全链路）
# ---------------------------------------------------------------------------
def answer_question(question: str,
                    kb: Any,
                    persona: str = "",
                    config: Optional[Dict[str, Any]] = None,
                    provider: Optional[str] = None) -> RagResult:
    """
    RAG 回答主流程：
      1) 清洗问题（去多余空格换行 + 超长截断）
      2) 问题加 query： 前缀 → 384 维向量
      3) Chroma 余弦召回 Top10
      4) 空召回兜底：直接返回「知识库未检索到相关资料」并终止
      5) Top10 与问题两两配对送入 bge-reranker-base
      6) logit → Sigmoid → 0~1 分数
      7) 降序重排取 Top3
      8) 置信度 = Top3 平均分 × 100，保留一位小数
      9) 低置信兜底：平均分 < 0.4 追加提示
     10) 「人设 + 用户问题 + Top3 片段」组装 prompt 调用 LLM
     11) 回答末尾追加置信度文本与提示说明

    配置来源（修复点 18）：每次都从磁盘 Configs/config.json 读取最新配置（含 API Key），
    参数 config 仅为兼容保留，传入启动时的旧快照也不会影响结果。
    注意：kb / llm_client 采用函数内导入，避免与 reranker 形成循环依赖。
    """
    # 延迟导入，规避 score_eval ←→ reranker/chroma_kb 的模块循环
    from lib import llm_client

    started = time.time()
    # 【修复点 18｜禁止模式 2】每次问答都重新读磁盘 Configs/config.json，
    # 不使用 app.py 启动时读入的旧 config 快照（否则管理员改完密钥/供应商后不生效）。
    # 入参 config 仅作兼容保留，不再作为数据源。
    cfg = load_config(force=True)
    rag = get_rag_params(cfg)
    result = RagResult(question=question, provider=provider or "")

    # 1) 问题清洗与截断保护
    clean = clean_question(question, max_chars=rag["max_question_chars"])
    result.question = clean
    if not clean:
        result.answer = EMPTY_QUESTION_TEXT
        result.no_result = True
        result.error = EMPTY_QUESTION_TEXT
        return result

    # 2~7) 召回 + 重排（由 ChromaKB.retrieve 封装：query 前缀 → 余弦 Top10 → Reranker → Top3）
    try:
        retrieval = kb.retrieve(clean)
    except Exception as exc:
        logger.error("检索阶段失败: %s", exc)
        result.error = f"检索失败：{exc}"
        result.answer = f"⚠️ {result.error}"
        result.elapsed_ms = int((time.time() - started) * 1000)
        return result

    result.recalled = retrieval.get("recalled", [])
    result.used = retrieval.get("used", [])
    result.rerank_ok = bool(retrieval.get("rerank_ok", True))

    # 4) 空召回兜底：终止流程，不调用 LLM
    if retrieval.get("empty") or not result.used:
        result.no_result = True
        result.answer = NO_RESULT_TEXT
        result.confidence_text = f"{CONFIDENCE_LABEL}：0.0%"
        result.elapsed_ms = int((time.time() - started) * 1000)
        logger.info("空召回兜底：问题「%s」未检索到资料", clean[:50])
        return result

    # 8) 置信度计算
    # 【需求2】改用 Top3 片段的**原始余弦相似度**：<0.5 的不参与平均；
    #          若全部 <0.5 则置信度记 0 并给出提示文案（见 compute_confidence）
    conf = compute_confidence(result.used)
    result.confidence = conf["confidence"]
    result.confidence_avg = conf["avg"]
    result.confidence_text = conf["text"]
    result.tips = conf.get("tips", "")
    result.low_confidence = conf["low_confidence"]
    result.provider = provider or llm_client.get_active_provider(cfg)

    # 10) 组装 prompt 并调用 LLM
    ok, answer = llm_client.answer_with_context(clean, result.used, persona=persona,
                                                provider=result.provider, config=cfg)
    if ok:
        result.llm_answer = answer
    else:
        # LLM 异常兜底：给出友好提示 + 检索资料摘要，不抛堆栈
        result.error = answer
        preview = "\n\n".join(
            f"《{c.get('source', 'unknown')}》片段：{(c.get('text') or '')[:120]}…"
            for c in result.used
        )
        result.llm_answer = f"⚠️ {answer}\n\n{LLM_FAIL_TIP}\n{preview}"

    # 11) 回答末尾追加置信度与提示说明
    result.answer = build_final_answer(result.llm_answer, conf)

    result.elapsed_ms = int((time.time() - started) * 1000)
    logger.info("问答完成：置信度 %.1f%%，耗时 %dms，召回 %d 条，使用 %d 条",
                result.confidence, result.elapsed_ms, len(result.recalled), len(result.used))
    return result


def build_final_answer(llm_answer: str, conf: Dict[str, Any]) -> str:
    """组装最终回答：正文 + 置信度 + 低置信提示。"""
    parts = [(llm_answer or "").strip()]
    parts.append("---")
    parts.append(f"📊 {conf['text']}")
    if conf.get("low_confidence"):
        parts.append(f"⚠️ {LOW_CONFIDENCE_TEXT}")
    return "\n\n".join(p for p in parts if p)


def build_debug_markdown(result: RagResult) -> str:
    """
    生成调试面板 Markdown：召回 Top10 片段与 Reranker 分数。
    （纯展示逻辑，便于管理员核对检索质量）
    """
    lines: List[str] = []
    lines.append(f"**问题**：{result.question or '（空）'}")
    lines.append(f"**耗时**：{result.elapsed_ms} ms ｜ **供应商**：{result.provider or '未调用'}"
                 f" ｜ **Reranker**：{'正常' if result.rerank_ok else '异常（已降级为向量相似度）'}")
    if result.no_result:
        lines.append("- 未检索到任何资料，已触发空召回兜底，未调用 LLM。")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"**召回 Top{len(result.recalled)}（含 Reranker 分数）**")
    lines.append("")
    lines.append("| 重排排名 | 召回排名 | Reranker 分数 | 余弦相似度 | 来源文件 | 片段摘要 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for c in result.recalled:
        summary = (c.get("text") or "").replace("|", "｜").replace("\n", " ")[:60]
        lines.append(
            f"| {c.get('rerank_rank', '-')} | {c.get('recall_rank', '-')} | "
            f"{float(c.get('rerank_score', 0)):.4f} | {float(c.get('similarity', 0)):.4f} | "
            f"{c.get('source', 'unknown')} | {summary}… |"
        )
    lines.append("")
    lines.append(f"**送入 LLM 的 Top{len(result.used)}**：" +
                 "、".join(f"《{c.get('source')}》#{c.get('chunk_index')}" for c in result.used))
    lines.append("")
    lines.append(f"**置信度**：{result.confidence:.1f}%（平均分 {result.confidence_avg:.4f}）"
                 + ("，已触发低置信兜底提示" if result.low_confidence else ""))
    return "\n".join(lines)
