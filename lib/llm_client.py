# -*- coding: utf-8 -*-
"""
文件名：lib/llm_client.py
功能：大模型 API 调用层
      1) DeepSeek / GML(智谱 GLM) 两个 OpenAI 兼容接口的调用封装
      2) 连通测试：保存 API Key 前调用一次极简 prompt 验证可用性
      3) RAG prompt 组装（人设 + Top3 参考资料）
      4) 完整异常兜底：密钥失效、网络超时、限流、服务端错误均返回友好文案，不抛堆栈
说明：API Key 从 Configs/config.json 读取（密文存储），代码中不存在任何硬编码密钥。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from lib import security
from lib.security import ConfigWriteError, get_api_key, load_config

logger = logging.getLogger(__name__)

# 支持的供应商（键名与 config.json 中 api 节点一致）
PROVIDERS = ("deepseek", "gml")
PROVIDER_LABELS = {"deepseek": "DeepSeek-API", "gml": "GML-API"}

# 供应商缺省参数（config.json 缺失时使用）
PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "path": "/chat/completions",
        "model": "deepseek-chat",
    },
    "gml": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "path": "/chat/completions",
        "model": "glm-4-flash",
    },
}

# 友好提示文案
MSG_NO_KEY = "未配置 API Key，请管理员在「更改 API-KEY」中配置后再试"
MSG_TIMEOUT = "调用大模型超时（网络较慢或服务繁忙），请稍后重试"
MSG_UNREACHABLE = "无法连接大模型服务（网络不可达或代理异常），请检查服务器网络"
MSG_INVALID_KEY = "API Key 无效或已失效，请管理员重新配置"
MSG_RATE_LIMIT = "大模型服务限流或余额不足，请稍后重试"
MSG_SERVER_ERROR = "大模型服务端异常，请稍后重试"
MSG_BAD_RESPONSE = "大模型返回内容异常，请稍后重试"


class LLMError(RuntimeError):
    """LLM 调用失败（携带已翻译的友好提示）。"""


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 配置读取（统一以磁盘为权威数据源）
# ---------------------------------------------------------------------------
def authoritative_config(config: Optional[Dict[str, Any]] = None, reason: str = "") -> Dict[str, Any]:
    """
    【修复点 9｜禁止模式 2｜最关键】获取“权威配置”。

    永远重新读磁盘 Configs/config.json，不信任调用方传入的旧内存快照
    （app.py 启动时读入的全局 CONFIG 属于快照，管理员改密钥后它不会自动更新）。
    如果发现调用方快照与磁盘不一致，打印差异日志，便于定位“内存与磁盘不一致”类问题。
    """
    fresh = security.load_config(force=True)          # ← 每次都读磁盘
    if config is not None:
        try:
            mem_api = (config.get("api", {}) or {})
            disk_api = (fresh.get("api", {}) or {})
            for p in PROVIDERS:
                mem_key = str((mem_api.get(p, {}) or {}).get("api_key", "") or "")
                disk_key = str((disk_api.get(p, {}) or {}).get("api_key", "") or "")
                if mem_key != disk_key:
                    logger.warning(
                        "[配置同步] 调用方内存快照与磁盘配置不一致（provider=%s，%s）："
                        "内存密文长度=%d，磁盘密文长度=%d —— 本次以磁盘配置为准",
                        p, reason or "LLM 调用", len(mem_key), len(disk_key))
        except Exception as exc:
            logger.debug("配置差异检查跳过: %s", exc)
    elif reason:
        logger.debug("[配置同步] %s：已从磁盘重新读取配置", reason)
    return fresh


def get_provider_config(provider: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """读取供应商连接参数（base_url / path / model），缺失项用缺省值补齐；配置取磁盘最新值。"""
    cfg = authoritative_config(config, reason="get_provider_config")
    node = dict(PROVIDER_DEFAULTS.get(provider, {}))
    node.update({k: v for k, v in (cfg.get("api", {}).get(provider, {}) or {}).items()
                 if k in ("base_url", "path", "model") and v})
    node.setdefault("base_url", PROVIDER_DEFAULTS.get(provider, {}).get("base_url", ""))
    node.setdefault("path", "/chat/completions")
    node.setdefault("model", PROVIDER_DEFAULTS.get(provider, {}).get("model", ""))
    return node


def get_llm_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取 LLM 调用参数（温度、最大 token、超时、重试次数）；配置取磁盘最新值。"""
    cfg = authoritative_config(config, reason="get_llm_params")
    node = cfg.get("llm", {}) or {}
    return {
        "temperature": float(node.get("temperature", 0.6)),
        "max_tokens": int(node.get("max_tokens", 1024)),
        "timeout": max(5, int(node.get("timeout", 90))),
        "max_retries": max(0, min(3, int(node.get("max_retries", 1)))),
    }


def get_active_provider(config: Optional[Dict[str, Any]] = None) -> str:
    """返回当前启用的供应商（默认 deepseek）；配置取磁盘最新值。"""
    cfg = authoritative_config(config, reason="get_active_provider")
    provider = str(cfg.get("active_provider", "deepseek")).lower()
    return provider if provider in PROVIDERS else "deepseek"


def set_active_provider(provider: str) -> str:
    """切换当前启用的供应商（保存到 config.json 并立即回读磁盘校验）。"""
    if provider not in PROVIDERS:
        raise LLMError(f"不支持的供应商：{provider}")
    fresh = security.update_config({"active_provider": provider})   # 写盘 + 回读
    actual = str(fresh.get("active_provider", "")).lower()
    if actual != provider:
        raise ConfigWriteError(f"切换供应商失败：磁盘上的值为 {actual or '空'}")
    logger.info("[配置写入] 当前启用供应商已切换为 %s（已回读磁盘校验）", provider)
    return actual


# ---------------------------------------------------------------------------
# 底层 HTTP 调用
# ---------------------------------------------------------------------------
def _build_url(cfg: Dict[str, str]) -> str:
    """拼接完整的 chat/completions 地址。"""
    base = (cfg.get("base_url") or "").rstrip("/")
    path = cfg.get("path") or "/chat/completions"
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _friendly_http_error(status_code: int, body: str = "") -> str:
    """把 HTTP 状态码翻译成友好提示。"""
    if status_code in (401, 403):
        return MSG_INVALID_KEY
    if status_code == 402:
        return MSG_RATE_LIMIT
    if status_code == 404:
        return "大模型接口地址或模型名不存在（404），请检查配置"
    if status_code == 429:
        return MSG_RATE_LIMIT
    if 500 <= status_code < 600:
        return MSG_SERVER_ERROR
    return f"大模型接口返回异常状态码 {status_code}"


def chat_completion(provider: str,
                    messages: List[Dict[str, str]],
                    api_key: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    timeout: Optional[int] = None) -> str:
    """
    调用 OpenAI 兼容的 chat/completions 接口，返回回答文本。
    失败时抛出 LLMError，其 message 已经是可直接展示给用户的友好文案。

    【修复点 10｜禁止模式 2｜最关键】api_key 未显式传入时，**必须调用
    security.get_api_key() 从磁盘读取密钥**（force=True），禁止复用程序启动时
    读入的旧全局变量，否则管理员保存密钥后仍会提示“未配置 API Key”。
    """
    cfg = authoritative_config(config, reason="chat_completion")
    if provider not in PROVIDERS:
        raise LLMError(f"不支持的供应商：{provider}")

    if api_key is not None and str(api_key).strip():
        key = str(api_key).strip()
        key_source = "调用方传入（连通测试临时密钥）"
    else:
        key = security.get_api_key(provider, force=True)     # ← 每次读磁盘最新配置
        key_source = "磁盘配置 Configs/config.json"

    if not key:
        logger.warning("[LLM 调用] provider=%s 无可用 API Key（来源=%s），返回未配置提示",
                       provider, key_source)
        raise LLMError(MSG_NO_KEY)                            # 未配置 → 前端友好提示
    logger.info("[LLM 调用] provider=%s 密钥来源=%s 密钥长度=%d", provider, key_source, len(key))

    node = get_provider_config(provider, cfg)
    params = get_llm_params(cfg)
    payload = {
        "model": node["model"],
        "messages": messages,
        "temperature": params["temperature"] if temperature is None else float(temperature),
        "max_tokens": params["max_tokens"] if max_tokens is None else int(max_tokens),
        "stream": False,
    }
    url = _build_url(node)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    wait = params["timeout"] if timeout is None else int(timeout)

    last_error = MSG_SERVER_ERROR
    for attempt in range(params["max_retries"] + 1):
        started = time.time()
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload),
                                 timeout=(10, wait))
            elapsed = time.time() - started
            if resp.status_code != 200:
                last_error = _friendly_http_error(resp.status_code, resp.text[:300])
                logger.error("LLM 调用失败 [%s] status=%s body=%s", provider, resp.status_code, resp.text[:300])
                # 密钥/参数类错误无需重试
                if resp.status_code in (400, 401, 403, 404):
                    raise LLMError(last_error)
                if attempt < params["max_retries"]:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMError(last_error)

            data = resp.json()
            text = _extract_content(data)
            logger.info("LLM 调用成功 [%s/%s]，耗时 %.2fs，回答 %d 字", provider, node["model"], elapsed, len(text))
            return text

        except requests.exceptions.Timeout:
            last_error = MSG_TIMEOUT
            logger.error("LLM 调用超时 [%s]，等待 %ss", provider, wait)
        except requests.exceptions.ConnectionError as exc:
            last_error = MSG_UNREACHABLE
            logger.error("LLM 网络不可达 [%s]: %s", provider, exc)
        except requests.exceptions.RequestException as exc:
            last_error = MSG_UNREACHABLE
            logger.error("LLM 请求异常 [%s]: %s", provider, exc)
        except ValueError as exc:      # JSON 解析失败
            last_error = MSG_BAD_RESPONSE
            logger.error("LLM 返回体解析失败 [%s]: %s", provider, exc)
        except LLMError:
            raise

        if attempt < params["max_retries"]:
            time.sleep(1.0 * (attempt + 1))

    raise LLMError(last_error)


def _extract_content(data: Dict[str, Any]) -> str:
    """从响应 JSON 中提取回答文本（兼容 OpenAI / 智谱格式）。"""
    try:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("响应中没有 choices 字段")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):    # 部分模型返回分段内容
            content = "".join(str(seg.get("text", "")) if isinstance(seg, dict) else str(seg) for seg in content)
        content = (content or "").strip()
        if not content:
            # 推理类模型可能把结果放在 reasoning_content
            content = (message.get("reasoning_content") or "").strip()
        if not content:
            raise ValueError("回答内容为空")
        return content
    except Exception as exc:
        raise ValueError(f"响应格式异常：{exc}") from exc


# ---------------------------------------------------------------------------
# 连通测试（管理员保存 API Key 前必做）
# ---------------------------------------------------------------------------
def test_connection(provider: str,
                    api_key: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    第一段连通测试：验证一个“待保存的候选密钥”能否调通模型。
    - 传入 api_key 时：测的是 UI 输入框里的临时密钥（测试通过才允许保存，此时尚未落盘）
    - 未传入 api_key 时：自动改为读取磁盘配置里的密钥再测（等价于 test_saved_connection）
    返回 (是否成功, 提示信息)。任何异常都被转换为友好文案。
    """
    if provider not in PROVIDERS:
        return False, f"不支持的供应商：{provider}"

    if api_key is not None and str(api_key).strip():
        key = str(api_key).strip()
        key_source = "临时输入密钥"
    else:
        # 【修复点 11】没传密钥就读磁盘，禁止拿内存旧值测试
        key = security.get_api_key(provider, force=True)
        key_source = "磁盘配置"
        if not key:
            return False, MSG_NO_KEY

    node = get_provider_config(provider, config)
    messages = [
        {"role": "system", "content": "你是一个连通性测试助手，只回复用户要求的字符。"},
        {"role": "user", "content": "请只回复两个字：正常"},
    ]
    started = time.time()
    try:
        text = chat_completion(provider, messages, api_key=key, config=config,
                               temperature=0.0, max_tokens=16, timeout=30)
        elapsed = time.time() - started
        preview = text.replace("\n", " ")[:40]
        logger.info("[连通测试] provider=%s 来源=%s 密钥长度=%d 耗时=%.2fs 结果=成功",
                    provider, key_source, len(key), elapsed)
        return True, (f"连通成功（{PROVIDER_LABELS.get(provider, provider)} / {node['model']}，"
                      f"密钥来源：{key_source}，长度 {len(key)}），耗时 {elapsed:.2f}s，返回：{preview}")
    except LLMError as exc:
        logger.warning("[连通测试] provider=%s 来源=%s 失败：%s", provider, key_source, exc)
        return False, f"连通测试失败：{exc}"
    except Exception as exc:  # 兜底：绝不把堆栈抛给界面
        logger.error("连通测试异常 [%s]: %s", provider, exc)
        return False, f"连通测试失败：{exc}"


def test_saved_connection(provider: str,
                          config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    第二段连通测试（二次校验）：

    【修复点 12｜禁止模式 1】只读取**磁盘上刚刚保存的配置**里的密钥发起调用，
    调用成功才判定为“真正连通”。彻底避免
    “拿前端临时输入key测试成功、但磁盘写入失败/后续读不到，却返回虚假成功”的问题。
    """
    if provider not in PROVIDERS:
        return False, f"不支持的供应商：{provider}"

    # 强制重新读盘，拿到磁盘上的密文与最新 base_url/model
    cfg = security.load_config(force=True)
    node = get_provider_config(provider, cfg)
    node_cfg = (cfg.get("api", {}) or {}).get(provider, {}) or {}
    cipher_len = len(str(node_cfg.get("api_key", "") or ""))

    key = security.get_api_key(provider, force=True)     # ← 从磁盘读，非内存快照
    if not key:
        logger.warning("[二次校验] provider=%s 磁盘配置中无密钥（密文长度=%d）", provider, cipher_len)
        return False, (f"未配置 API Key：磁盘 {security.CONFIG_PATH.name} 中未找到 {provider} 的可用密钥，"
                       f"请保存后再试")

    messages = [
        {"role": "system", "content": "你是一个连通性测试助手，只回复用户要求的字符。"},
        {"role": "user", "content": "请只回复两个字：正常"},
    ]
    started = time.time()
    try:
        text = chat_completion(provider, messages, api_key=key, config=cfg,
                               temperature=0.0, max_tokens=16, timeout=30)
        elapsed = time.time() - started
        preview = text.replace("\n", " ")[:40]
        logger.info("[二次校验] provider=%s 磁盘密钥长度=%d 模型=%s 耗时=%.2fs 结果=成功",
                    provider, len(key), node["model"], elapsed)
        return True, (f"已连接（二次校验通过：读取磁盘配置，密钥长度 {len(key)}，"
                      f"模型 {node['model']}），耗时 {elapsed:.2f}s，返回：{preview}")
    except LLMError as exc:
        logger.error("[二次校验] provider=%s 磁盘密钥调用失败：%s", provider, exc)
        return False, f"磁盘配置中的密钥调用失败：{exc}"
    except Exception as exc:
        logger.error("[二次校验] provider=%s 异常：%s", provider, exc)
        return False, f"磁盘配置中的密钥调用异常：{exc}"


def save_and_verify_api_key(provider: str,
                            api_key: str,
                            set_active: bool = False,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    管理员“更改 API-KEY”的完整流程（两段式，缺一不可）：

      1) 用前端传入的临时密钥做一次连通测试（通过才允许保存）
      2) 写入磁盘 Configs/config.json（with + flush + fsync，密文存储）
      3) **立即回读磁盘配置，用磁盘上的密钥再调用一次（二次校验）**
      4) 二次校验通过 → 判定真正成功；失败 → 回滚到保存前的密钥并报错

    返回 dict：ok / provider / stage / message / first_test / second_test /
              key_length / active_provider / config

    【验证点 1】管理员点击「连通测试并保存」后，两段都通过才返回 ok=True：
      - 第 1 段用输入框里的临时密钥测试（此前的错误实现到此就返回“成功”了 → 禁止模式 1）
      - 第 2 段保存到磁盘后**重新读磁盘配置**再测一次，才是最终结论
    """
    result: Dict[str, Any] = {
        "ok": False, "provider": provider, "stage": "", "message": "",
        "first_test": "", "second_test": "", "key_length": 0,
        "active_provider": "", "config": None,
    }
    if provider not in PROVIDERS:
        result.update(stage="validate", message=f"不支持的供应商：{provider}")
        return result
    key = (api_key or "").strip()
    if not key:
        result.update(stage="validate", message="请填写新的 API Key")
        return result
    result["key_length"] = len(key)

    # ---- 第 1 段：临时密钥连通测试（不落盘）----
    ok1, msg1 = test_connection(provider, api_key=key, config=config)
    result["first_test"] = msg1
    if not ok1:
        result.update(stage="first_test",
                      message=f"❌ {msg1}\n\n**未保存**：临时密钥连通测试未通过，磁盘配置保持原样。")
        return result

    # ---- 第 2 段：保存到磁盘 + 回读磁盘做二次校验 ----
    old_cipher = security.get_api_key_cipher(provider)      # 备份，用于失败回滚
    try:
        security.set_api_key(provider, key)                 # 写盘 + 回读校验（内部已 flush+fsync）
    except (ConfigWriteError, OSError) as exc:
        result.update(stage="save", message=f"❌ 保存失败：{exc}\n\n磁盘配置保持原样。")
        return result

    ok2, msg2 = test_saved_connection(provider, config=config)
    result["second_test"] = msg2
    if not ok2:
        # 二次校验失败：回滚，绝不留下“看似保存成功、实际不可用”的密钥
        try:
            security.set_api_key_cipher(provider, old_cipher)
            rollback_tip = "已回滚为保存前的密钥。"
        except Exception as exc:
            rollback_tip = f"⚠️ 回滚失败：{exc}"
        result.update(stage="second_test",
                      message=(f"❌ 二次校验失败（读取磁盘配置后调用失败）：{msg2}\n\n"
                               f"{rollback_tip}\n请检查网络/密钥权限后重试。"))
        logger.error("[保存密钥] provider=%s 二次校验失败，已回滚", provider)
        return result

    # ---- 可选：把该供应商设为当前启用的大模型（写盘 + 回读校验）----
    active = get_active_provider()
    if set_active:
        try:
            active = set_active_provider(provider)
        except Exception as exc:
            logger.error("[保存密钥] 切换供应商失败：%s", exc)
            result["message"] = f"⚠️ 密钥已保存，但切换当前供应商失败：{exc}"
    fresh = security.load_config(force=True)                # 最终以磁盘内容为准
    result.update(ok=True, stage="done", active_provider=active, config=fresh,
                  message=(f"✅ {msg1}\n✅ {msg2}\n\n"
                           f"已加密保存 {PROVIDER_LABELS.get(provider, provider)}"
                           f"（密钥长度 {len(key)}），当前启用供应商：`{active}`"))
    logger.info("[保存密钥] provider=%s 两段式校验全部通过，启用供应商=%s", provider, active)
    return result


def test_all_providers(config: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[bool, str]]:
    """批量测试两个供应商**磁盘配置中密钥**的连通性（用于状态面板刷新）。"""
    return {p: test_saved_connection(p, config=config) for p in PROVIDERS}


# ---------------------------------------------------------------------------
# RAG Prompt 组装
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "你是本地知识库助手，只能依据【参考资料】回答用户问题。"
    "资料中没有的信息必须明确说明“知识库资料中未提及”，不得编造。"
)


def build_context_block(chunks: List[Dict[str, Any]], max_context_chars: int = 6000) -> str:
    """
    把 Top-K 片段拼成参考资料块：
        [资料1] 来源：《文件名》
        内容：...
    长度保护：
      - 累加后超出 max_context_chars 时，停止追加后续片段（保证已收录资料完整）；
      - 若单条片段本身就超出上限（例如切片参数被调大），对该条做硬截断，
        避免超长 prompt 导致大模型报错。
    """
    blocks: List[str] = []
    used = 0
    for i, chunk in enumerate(chunks or [], start=1):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        source = chunk.get("source", "unknown")
        block = f"[资料{i}] 来源：《{source}》\n内容：{text}"
        remaining = max_context_chars - used
        if remaining <= 0:
            logger.warning("参考资料已达上限 %d 字符，忽略后续片段", max_context_chars)
            break
        if len(block) > remaining:
            if blocks:                      # 已有资料：保持完整性，直接停止
                break
            block = block[:remaining]       # 单条即超限：硬截断
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_rag_messages(question: str,
                       chunks: List[Dict[str, Any]],
                       persona: str = "",
                       max_context_chars: int = 6000) -> List[Dict[str, str]]:
    """组装最终 prompt：系统人设 + 【参考资料】 + 用户问题。"""
    system_prompt = (persona or "").strip() or DEFAULT_SYSTEM_PROMPT
    context = build_context_block(chunks, max_context_chars=max_context_chars)
    system_content = (
        f"{system_prompt}\n\n"
        "【参考资料】（以下内容来自本地知识库，请严格依据这些资料作答）\n"
        f"{context if context else '（无）'}\n\n"
        "【作答要求】\n"
        "1. 先给结论，再给依据；引用资料时标注《文件名》。\n"
        "2. 资料未覆盖的内容直接说明“知识库资料中未提及”。\n"
        "3. 不要在回答中输出置信度数值，置信度由系统在末尾统一标注。"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": (question or "").strip()},
    ]


def answer_with_context(question: str,
                        chunks: List[Dict[str, Any]],
                        persona: str = "",
                        provider: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    组装 prompt 并调用 LLM 生成最终回答。
    返回 (是否成功, 回答文本或友好错误提示)。

    【修复点 13｜禁止模式 2】RAG 问答链路每次都重新读磁盘配置取密钥与供应商，
    不使用程序启动时的旧全局变量，保证管理员改完密钥后对话立即生效。
    """
    cfg = authoritative_config(config, reason="answer_with_context")
    rag = cfg.get("rag", {}) or {}
    provider = provider or get_active_provider(cfg)
    messages = build_rag_messages(question, chunks, persona,
                                  max_context_chars=int(rag.get("max_context_chars", 6000)))
    try:
        return True, chat_completion(provider, messages, config=cfg)   # 密钥由 chat_completion 读磁盘
    except LLMError as exc:
        return False, str(exc)
    except Exception as exc:  # 兜底
        logger.error("调用 LLM 出现未预期异常: %s", exc)
        return False, f"调用大模型失败：{exc}"
