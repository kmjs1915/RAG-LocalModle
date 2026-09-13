# -*- coding: utf-8 -*-
"""
文件名：lib/security.py
功能：安全与配置文件层
      1) bcrypt 密码哈希与校验、密码强度校验
      2) Configs/config.json 读写（应用配置 + RAG 参数）
      3) Configs/security.json 读写（管理员账号 + 密码哈希）
      4) API Key 密文存储（Fernet 对称加密，密钥保存在 Configs/.secret.key）
      5) Database/access_log.jsonl 访问日志写入与聚合（仅记录提问行为）
说明：本模块为纯业务逻辑，不包含任何 UI 代码。
"""

from __future__ import annotations

import base64
import copy
import getpass
import hashlib
import hmac
import json
import logging
import os
import platform
import secrets
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bcrypt

logger = logging.getLogger(__name__)


class ConfigWriteError(RuntimeError):
    """配置文件写入失败（IO / 序列化 / 落盘校验不通过）。"""


class ConfigStaleError(RuntimeError):
    """内存中的配置与磁盘不一致（说明调用方使用了过期的旧快照）。"""

# ---------------------------------------------------------------------------
# 路径常量（全部基于相对路径，适配 Windows）
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "Configs"
CONFIG_PATH = CONFIGS_DIR / "config.json"
SECURITY_PATH = CONFIGS_DIR / "security.json"
SECRET_KEY_PATH = CONFIGS_DIR / ".secret.key"      # API Key 加密主密钥（不入库、不明文写配置）
DATABASE_DIR = ROOT_DIR / "Database"
ACCESS_LOG_PATH = DATABASE_DIR / "access_log.jsonl"
APP_LOG_PATH = DATABASE_DIR / "app.log"
USER_DOCS_DIR = ROOT_DIR / "user_docs"
CHROMA_DIR = ROOT_DIR / "chroma_db"
PERSONA_PATH = ROOT_DIR / "agent_config" / "Person.txt"

# 默认管理员账号（首次运行自动写入 security.json，密码仅存 bcrypt 哈希）
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "abc123456"

# 配置文件读写的进程内互斥锁，避免多线程同时写入造成 JSON 截断
_FILE_LOCK = threading.RLock()
# 访问日志写入锁
_LOG_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """确保运行所需目录存在（首次运行或目录被误删时自动重建）。"""
    for d in (CONFIGS_DIR, DATABASE_DIR, USER_DOCS_DIR, CHROMA_DIR, ROOT_DIR / "agent_config"):
        d.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    """返回可读时间字符串，用于日志与界面展示。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write_text(path: Path, text: str) -> None:
    """
    原子写文件：先写临时文件，flush + fsync 落盘，再原子替换，避免写入中断导致配置损坏。

    【修复点 1｜禁止模式 3】必须使用 with open(...) 上下文管理器：
    - with 退出时自动 close()，close 前会 flush 用户态缓冲区；
    - 额外显式调用 fh.flush() + os.fsync(fh.fileno())，强制操作系统把数据真正写到磁盘，
      避免“缓冲区未落地、磁盘文件实际没有更新”的情况；
    - IO 异常统一捕获并抛出 ConfigWriteError，由上层转成界面提示。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:   # ← with 上下文管理器
            fh.write(text)
            fh.flush()                    # 刷新 Python 缓冲区到操作系统
            os.fsync(fh.fileno())         # 刷新操作系统缓冲区到磁盘（确保真正落地）
        os.replace(tmp, path)             # 原子替换，避免读到半截文件
    except OSError as exc:                # 磁盘满 / 权限不足 / 文件被占用
        logger.error("写入文件失败（IO 异常）%s: %s", path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise ConfigWriteError(f"写入配置失败（IO 异常）：{exc}") from exc


def _read_json(path: Path) -> Dict[str, Any]:
    """
    读取 JSON 文件；文件不存在或损坏时返回空字典（不抛异常）。

    【修复点 2｜边界处理】显式 with 上下文读取，分别捕获 IO 异常与 JSON 解析异常，
    并在控制台打印可诊断信息（文件路径、字节数、异常原因），不打印文件内容。
    """
    if not path.exists():
        logger.warning("[配置读取] 文件不存在：%s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:      # ← with 上下文管理器
            raw = fh.read()
    except OSError as exc:                                  # 权限 / 被占用 / 设备错误
        logger.error("[配置读取] IO 异常 %s: %s", path, exc)
        return {}
    except UnicodeDecodeError as exc:                        # 编码异常
        logger.error("[配置读取] 编码异常 %s: %s", path, exc)
        return {}
    if not raw.strip():
        logger.warning("[配置读取] 文件为空：%s", path)
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:                      # JSON 语法损坏
        logger.error("[配置读取] JSON 解析异常 %s（%d 字节）: 行%s 列%s %s",
                     path, len(raw.encode("utf-8")), exc.lineno, exc.colno, exc.msg)
        return {}
    if not isinstance(data, dict):
        logger.error("[配置读取] JSON 根节点不是对象，已忽略：%s", path)
        return {}
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """写 JSON 文件（缩进 + 保留中文）；序列化异常与 IO 异常均转为 ConfigWriteError。"""
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:      # 出现不可序列化对象
        logger.error("配置序列化失败 %s: %s", path, exc)
        raise ConfigWriteError(f"配置序列化失败：{exc}") from exc
    _atomic_write_text(path, text)


def _deep_merge(defaults: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """以 defaults 为模板补齐 current 缺失的键（用户配置升级用，不覆盖已有值）。"""
    result: Dict[str, Any] = {}
    for key, dv in defaults.items():
        if key in current:
            cv = current[key]
            if isinstance(dv, dict) and isinstance(cv, dict):
                result[key] = _deep_merge(dv, cv)
            else:
                result[key] = cv
        else:
            result[key] = dv
    # 保留用户额外添加的键
    for key, cv in current.items():
        if key not in result:
            result[key] = cv
    return result


# ---------------------------------------------------------------------------
# 默认配置模板
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 1,
    "api": {
        "deepseek": {
            "api_key": "",
            "base_url": "https://api.deepseek.com",
            "path": "/chat/completions",
            "model": "deepseek-chat",
        },
        "gml": {
            "api_key": "",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "path": "/chat/completions",
            "model": "glm-4-flash",
        },
    },
    "active_provider": "deepseek",
    # RAG 关键参数（切片、召回、重排、置信度阈值）
    "rag": {
        "chunk_size": 800,                 # 切片长度（字符）
        "chunk_overlap": 150,              # 切片重叠长度（字符）
        "top_k_recall": 10,                # 向量召回候选数
        "top_k_rerank": 3,                 # 重排后送入 LLM 的片段数
        "low_confidence_threshold": 0.4,   # 低置信度兜底阈值
        "max_question_chars": 2000,        # 用户问题最大长度（超长截断保护）
        "max_context_chars": 6000,         # 拼进 prompt 的参考资料最大字符数
    },
    "llm": {"temperature": 0.6, "max_tokens": 1024, "timeout": 90, "max_retries": 1},
    "embedding": {
        "model_path": "models/bge-small-zh-v1.5",
        "dim": 384,
        "query_prefix": "query：",         # 查询前缀（强制）
        "passage_prefix": "passage：",     # 文档切片前缀（强制）
        "max_length": 512,
        "batch_size": 32,
    },
    "reranker": {"model_path": "models/bge-reranker-base", "max_length": 512, "batch_size": 16},
    "chroma": {"persist_dir": "chroma_db", "collection": "personal_kb"},
    "server": {"server_name": "0.0.0.0", "server_port": 7860},
}


# ---------------------------------------------------------------------------
# 内存配置镜像（只作为“镜像/诊断”，磁盘始终是唯一权威数据源）
# ---------------------------------------------------------------------------
_CONFIG_SNAPSHOT: Dict[str, Any] = {}      # 最近一次从磁盘读到的配置副本
_CONFIG_MTIME: int = -1                    # 该副本对应文件的修改时间（纳秒）
_CONFIG_READ_COUNT: int = 0                # 读盘次数（用于诊断“是否每次都在读磁盘”）


def load_config(force: bool = True) -> Dict[str, Any]:
    """
    读取应用配置。缺失字段时自动补齐默认值并落盘。

    【修复点 3｜禁止模式 2】默认 force=True —— 每次都从磁盘 Configs/config.json 读取，
    不再返回进程启动时缓存下来的旧对象；读到后同步更新内存镜像 _CONFIG_SNAPSHOT，
    保证“内存 == 磁盘”。返回值是深拷贝，调用方随意修改也不会污染镜像。

    force=False 时走 mtime 缓存快照（仅用于高频只读场景，例如每轮对话里的多次读取）。
    """
    global _CONFIG_MTIME, _CONFIG_READ_COUNT
    with _FILE_LOCK:
        ensure_dirs()

        try:
            mtime = CONFIG_PATH.stat().st_mtime_ns
        except OSError:
            mtime = -1

        # 快速路径：文件没变且有镜像时直接复用（避免同一次请求内反复读盘）
        if not force and _CONFIG_SNAPSHOT and mtime == _CONFIG_MTIME:
            return copy.deepcopy(_CONFIG_SNAPSHOT)

        current = _read_json(CONFIG_PATH)                # ← 每次都真正读磁盘
        merged = _deep_merge(DEFAULT_CONFIG, current)
        if merged != current:
            # 首次运行或配置文件缺字段：补全后落盘
            try:
                _write_json(CONFIG_PATH, merged)
                logger.info("[配置写入] 已补齐默认字段并保存：%s", CONFIG_PATH)
            except ConfigWriteError as exc:
                logger.error("[配置写入] 补齐默认字段失败：%s", exc)

        _CONFIG_SNAPSHOT.clear()
        _CONFIG_SNAPSHOT.update(copy.deepcopy(merged))    # 同步内存镜像
        try:
            _CONFIG_MTIME = CONFIG_PATH.stat().st_mtime_ns
        except OSError:
            _CONFIG_MTIME = mtime
        _CONFIG_READ_COUNT += 1
        logger.debug("[配置读取] 第 %d 次读盘完成：%s", _CONFIG_READ_COUNT, CONFIG_PATH)
        return copy.deepcopy(merged)


def reload_config() -> Dict[str, Any]:
    """强制从磁盘重新加载配置并刷新内存镜像（供 UI 刷新、登录、写操作后调用）。"""
    return load_config(force=True)


def get_config_snapshot() -> Dict[str, Any]:
    """返回内存镜像副本（诊断用；业务逻辑请勿以此为准）。"""
    with _FILE_LOCK:
        return copy.deepcopy(_CONFIG_SNAPSHOT)


def config_read_count() -> int:
    """返回累计读盘次数（验证“每次调用都读磁盘”用）。"""
    return _CONFIG_READ_COUNT


def save_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    保存应用配置并**立即回读磁盘**，返回磁盘上的最新配置。

    【修复点 4｜禁止模式 2 / 3】写盘使用 with open + flush + fsync；
    写完后马上调用 load_config(force=True) 重新读盘刷新内存镜像，
    确保调用方拿到的一定是磁盘真实内容，内存与磁盘完全一致。
    """
    with _FILE_LOCK:
        ensure_dirs()
        _write_json(CONFIG_PATH, config)                  # 内部已 flush + fsync
        fresh = load_config(force=True)                   # ← 写后立即回读磁盘
        logger.info("[配置写入] 已保存并回读校验：%s", CONFIG_PATH)
        return fresh


def update_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    """按路径浅合并更新配置并保存，返回**磁盘上的最新配置**（写后回读）。"""
    with _FILE_LOCK:
        cfg = load_config(force=True)                     # 以磁盘为准，避免覆盖他人写入
        cfg = _deep_merge(cfg, patch)
        return save_config(cfg)


def get_rag_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """获取 RAG 核心参数（带类型与范围保护）。"""
    cfg = config or load_config()
    rag = cfg.get("rag", {})
    chunk_size = max(100, int(rag.get("chunk_size", 800)))
    chunk_overlap = int(rag.get("chunk_overlap", 150))
    # 重叠必须小于切片长度，否则会死循环
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k_recall": max(1, int(rag.get("top_k_recall", 10))),
        "top_k_rerank": max(1, int(rag.get("top_k_rerank", 3))),
        "low_confidence_threshold": float(rag.get("low_confidence_threshold", 0.4)),
        "max_question_chars": max(50, int(rag.get("max_question_chars", 2000))),
        "max_context_chars": max(500, int(rag.get("max_context_chars", 6000))),
    }


# ---------------------------------------------------------------------------
# 密码哈希（bcrypt）
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """使用 bcrypt 生成密码哈希（自动加盐，cost 默认 12）。"""
    if password is None:
        password = ""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """校验明文密码与 bcrypt 哈希是否匹配（异常一律视为不匹配）。"""
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception as exc:
        logger.error("密码校验异常: %s", exc)
        return False


def check_password_strength(password: str) -> Tuple[bool, str]:
    """
    新密码简单强度校验：
      长度 >= 8；同时包含字母与数字；不能与默认密码相同。
    返回 (是否通过, 提示信息)
    """
    if not password or len(password) < 8:
        return False, "新密码长度至少 8 位"
    has_alpha = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    if not (has_alpha and has_digit):
        return False, "新密码需同时包含字母和数字"
    if password == DEFAULT_ADMIN_PASSWORD:
        return False, "不允许继续使用默认密码"
    if password.strip() != password:
        return False, "新密码首尾不能包含空格"
    return True, "密码强度校验通过"


# ---------------------------------------------------------------------------
# 管理员账号（Configs/security.json）
# ---------------------------------------------------------------------------
def ensure_security_file() -> Dict[str, Any]:
    """
    确保 security.json 存在。
    - 首次运行：写入默认管理员账号 admin，密码仅保存 bcrypt 哈希
    - 已存在：仅补齐缺失字段，不覆盖已有哈希
    返回 security 配置字典。
    """
    with _FILE_LOCK:
        ensure_dirs()
        data = _read_json(SECURITY_PATH)
        changed = False
        if "admin" not in data or not isinstance(data.get("admin"), dict):
            data["admin"] = {}
            changed = True
        admin = data["admin"]
        if not admin.get("username"):
            admin["username"] = DEFAULT_ADMIN_USERNAME
            changed = True
        if not admin.get("password_hash"):
            # 仅在此处出现一次明文密码，用于生成哈希，落盘的是哈希值
            admin["password_hash"] = hash_password(DEFAULT_ADMIN_PASSWORD)
            admin["created_at"] = now_str()
            admin["is_default_password"] = True
            changed = True
        if "updated_at" not in admin:
            admin["updated_at"] = now_str()
            changed = True
        if "version" not in data:
            data["version"] = 1
            changed = True
        if changed:
            _write_json(SECURITY_PATH, data)
            logger.info("已初始化管理员配置文件: %s", SECURITY_PATH)
        return data


def load_security() -> Dict[str, Any]:
    """读取管理员安全配置。"""
    return ensure_security_file()


def verify_admin(username: str, password: str) -> Tuple[bool, str]:
    """
    校验管理员登录。返回 (是否通过, 提示信息)。
    - 账号或密码错误统一提示，避免泄露账号是否存在。
    """
    data = ensure_security_file()
    admin = data.get("admin", {})
    stored_user = admin.get("username", DEFAULT_ADMIN_USERNAME)
    stored_hash = admin.get("password_hash", "")
    username = (username or "").strip()
    if not username or not password:
        return False, "请输入管理员账号与密码"
    if username.lower() != str(stored_user).lower():
        return False, "账号或密码错误"
    if not verify_password(password, stored_hash):
        return False, "账号或密码错误"
    return True, "登录成功"


def change_admin_password(old_password: str, new_password: str, confirm_password: str) -> Tuple[bool, str]:
    """
    修改管理员密码：校验旧密码哈希、两次新密码一致、新密码强度。
    成功后写入 security.json（仅存 bcrypt 哈希）。
    """
    data = ensure_security_file()
    admin = data.get("admin", {})
    if not verify_password(old_password, admin.get("password_hash", "")):
        return False, "旧密码不正确"
    if not new_password or new_password != confirm_password:
        return False, "两次输入的新密码不一致"
    if new_password == old_password:
        return False, "新密码不能与旧密码相同"
    ok, msg = check_password_strength(new_password)
    if not ok:
        return False, msg
    admin["password_hash"] = hash_password(new_password)
    admin["updated_at"] = now_str()
    admin["is_default_password"] = new_password == DEFAULT_ADMIN_PASSWORD
    data["admin"] = admin
    with _FILE_LOCK:
        _write_json(SECURITY_PATH, data)
    logger.info("管理员密码已更新")
    return True, "密码修改成功，请牢记新密码"


def is_default_admin_password() -> bool:
    """判断当前是否仍在使用默认密码（用于启动提示）。"""
    try:
        data = load_security()
        return verify_password(DEFAULT_ADMIN_PASSWORD, data.get("admin", {}).get("password_hash", ""))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# API Key 密文存储
# ---------------------------------------------------------------------------
# 设计说明：
#   1) 优先使用 cryptography.Fernet（AES-128-CBC + HMAC）做对称加密；
#   2) 主密钥保存在 Configs/.secret.key（首次运行随机生成，权限尽量收紧），
#      config.json 中只保存密文，代码里不存在任何明文密钥；
#   3) 若环境缺少 cryptography，则退化为 PBKDF2-HMAC-SHA256 派生密钥流 + HMAC 校验，
#      保证功能可用（安全性弱于 Fernet，仅作兜底）。
_ENC_PREFIX = "enc:v1:"
_ENC_PREFIX_FALLBACK = "enc:v1x:"


def _machine_seed() -> str:
    """采集本机特征，参与密钥派生（换机器后密文无法解密，属预期行为）。"""
    parts = [
        platform.node(),
        platform.machine(),
        getpass.getuser() if hasattr(getpass, "getuser") else "",
        str(ROOT_DIR),
    ]
    return "|".join(parts)


def _load_or_create_master_key() -> bytes:
    """读取或生成主密钥（32 字节随机值，base64 存储）。"""
    with _FILE_LOCK:
        ensure_dirs()
        if SECRET_KEY_PATH.exists():
            try:
                raw = SECRET_KEY_PATH.read_text(encoding="utf-8").strip()
                if raw:
                    return base64.urlsafe_b64decode(raw.encode("ascii"))
            except Exception as exc:
                logger.error("读取主密钥失败，将重新生成: %s", exc)
        key = secrets.token_bytes(32)
        _atomic_write_text(SECRET_KEY_PATH, base64.urlsafe_b64encode(key).decode("ascii"))
        try:  # Windows 下尽量收紧文件权限（失败不影响功能）
            os.chmod(SECRET_KEY_PATH, 0o600)
        except Exception:
            pass
        logger.info("已生成 API Key 加密主密钥: %s", SECRET_KEY_PATH)
        return key


def _fallback_cipher(key: bytes, nonce: bytes, data: bytes) -> bytes:
    """兜底加解密：PBKDF2 派生密钥流与数据异或（同一函数可逆）。"""
    stream = hashlib.pbkdf2_hmac("sha256", key + _machine_seed().encode("utf-8"), nonce, 4096, dklen=len(data))
    return bytes(a ^ b for a, b in zip(data, stream))


def encrypt_secret(plain: str) -> str:
    """加密 API Key，返回可写入 config.json 的密文字符串。"""
    plain = plain or ""
    if not plain:
        return ""
    master = _load_or_create_master_key()
    try:
        from cryptography.fernet import Fernet  # type: ignore

        token = Fernet(base64.urlsafe_b64encode(master)).encrypt(plain.encode("utf-8")).decode("ascii")
        return _ENC_PREFIX + token
    except Exception as exc:
        logger.warning("Fernet 不可用，改用兜底加密: %s", exc)
        nonce = secrets.token_bytes(16)
        body = _fallback_cipher(master, nonce, plain.encode("utf-8"))
        sig = hmac.new(master, nonce + body, hashlib.sha256).digest()[:16]
        return _ENC_PREFIX_FALLBACK + base64.urlsafe_b64encode(nonce + sig + body).decode("ascii")


def decrypt_secret(cipher_text: str) -> str:
    """解密 API Key；失败返回空字符串（不抛异常）。"""
    if not cipher_text:
        return ""
    master = _load_or_create_master_key()
    try:
        if cipher_text.startswith(_ENC_PREFIX):
            from cryptography.fernet import Fernet  # type: ignore

            token = cipher_text[len(_ENC_PREFIX):].encode("ascii")
            return Fernet(base64.urlsafe_b64encode(master)).decrypt(token).decode("utf-8")
        if cipher_text.startswith(_ENC_PREFIX_FALLBACK):
            blob = base64.urlsafe_b64decode(cipher_text[len(_ENC_PREFIX_FALLBACK):].encode("ascii"))
            nonce, sig, body = blob[:16], blob[16:32], blob[32:]
            expect = hmac.new(master, nonce + body, hashlib.sha256).digest()[:16]
            if not hmac.compare_digest(sig, expect):
                logger.error("API Key 密文校验失败（可能被篡改）")
                return ""
            return _fallback_cipher(master, nonce, body).decode("utf-8")
    except Exception as exc:
        logger.error("API Key 解密失败: %s", exc)
        return ""
    # 兼容历史明文（极早期版本可能直接存明文），读取时按明文处理
    return cipher_text


def is_encrypted(text: str) -> bool:
    """判断字符串是否为本模块生成的密文。"""
    return bool(text) and (text.startswith(_ENC_PREFIX) or text.startswith(_ENC_PREFIX_FALLBACK))


def get_api_key(provider: str, config: Optional[Dict[str, Any]] = None, force: bool = True) -> str:
    """
    获取指定供应商的明文 API Key（仅在调用 LLM 时于内存中使用）。

    【修复点 5｜禁止模式 2｜最关键】默认 force=True：**每次调用都重新读磁盘**，
    不再信任调用方可能传入的、程序启动时读入的旧内存快照。
    这样管理员保存密钥后，正在运行的对话链路可以立刻读到新密钥。

    【修复点 6｜边界处理】控制台打印密钥长度与掩码，绝不打印明文；
    磁盘读不到密钥时返回空字符串，由上层给出“未配置 API Key”提示。
    """
    if force:
        cfg = load_config(force=True)                     # ← 以磁盘最新配置为唯一权威
        if config is not None:
            disk_cipher = ((cfg.get("api", {}) or {}).get(provider, {}) or {}).get("api_key", "")
            mem_cipher = ((config.get("api", {}) or {}).get(provider, {}) or {}).get("api_key", "")
            if disk_cipher != mem_cipher:
                logger.warning("[密钥读取] 检测到调用方内存快照已过期（provider=%s），"
                               "本次以磁盘配置为准", provider)
    else:
        cfg = config or load_config(force=False)

    node = (cfg.get("api", {}) or {}).get(provider, {}) or {}
    raw = node.get("api_key", "") or ""
    key = decrypt_secret(raw) if raw else ""

    if key:
        logger.info("[密钥读取] provider=%s 来源=磁盘配置 密钥长度=%d 掩码=%s",
                    provider, len(key), mask_api_key(key))
    else:
        logger.warning("[密钥读取] provider=%s 磁盘配置中未找到可用 API Key"
                       "（密文长度=%d，文件=%s）", provider, len(raw), CONFIG_PATH)
    return key


def get_api_key_cipher(provider: str) -> str:
    """读取指定供应商的密钥密文（用于保存失败时回滚）。"""
    cfg = load_config(force=True)
    return str(((cfg.get("api", {}) or {}).get(provider, {}) or {}).get("api_key", "") or "")


def set_api_key_cipher(provider: str, cipher_text: str) -> Dict[str, Any]:
    """直接写回密钥密文（回滚专用），写完立即回读磁盘并返回最新配置。"""
    with _FILE_LOCK:
        cfg = load_config(force=True)
        cfg.setdefault("api", {}).setdefault(provider, {})
        cfg["api"][provider]["api_key"] = cipher_text or ""
        cfg["api"][provider]["updated_at"] = now_str()
        return save_config(cfg)


def set_api_key(provider: str, api_key: str, verify: bool = True) -> Dict[str, Any]:
    """
    加密保存指定供应商的 API Key，返回磁盘上的最新配置。

    【修复点 4｜禁止模式 3】写入磁盘使用 with open + flush + fsync（见 _atomic_write_text）。
    【修复点 7｜禁止模式 2】保存成功后**立即重新从磁盘加载一次配置**，
      刷新本模块内存镜像，并校验“磁盘上能解出的密钥 == 本次要保存的密钥”，
      彻底避免“只改了局部临时变量、全局内存没更新”的问题。
    校验不通过时抛 ConfigWriteError，界面会提示保存失败（而不是虚假成功）。

    【验证点 4】密钥落盘后，重启 Python 服务仍能读到：
      写入用 with open + flush + os.fsync 保证真正落盘，读盘统一走 load_config(force=True)。
    """
    key = (api_key or "").strip()
    with _FILE_LOCK:
        cfg = load_config(force=True)                     # 以磁盘为准，避免覆盖他人改动
        cfg.setdefault("api", {}).setdefault(provider, {})
        cfg["api"][provider]["api_key"] = encrypt_secret(key)
        cfg["api"][provider]["updated_at"] = now_str()

        save_config(cfg)                                  # 写盘（with + flush + fsync）

        # ---- 写后立即回读磁盘，刷新内存并做落盘校验 ----
        fresh = load_config(force=True)
        disk_raw = ((fresh.get("api", {}) or {}).get(provider, {}) or {}).get("api_key", "")
        disk_key = decrypt_secret(disk_raw) if disk_raw else ""

        if verify and disk_key != key:
            logger.error("[密钥写入] 落盘校验失败 provider=%s：期望长度=%d，磁盘回读长度=%d",
                         provider, len(key), len(disk_key))
            raise ConfigWriteError(
                f"API Key 落盘校验失败（期望长度 {len(key)}，磁盘回读长度 {len(disk_key)}），"
                f"请检查 {CONFIG_PATH} 的写入权限"
            )

        logger.info("[密钥写入] 已加密保存并回读校验通过：provider=%s 密钥长度=%d 掩码=%s 文件=%s",
                    provider, len(disk_key), mask_api_key(disk_key) if disk_key else "（空）", CONFIG_PATH)
        return fresh


def clear_api_key(provider: str) -> Dict[str, Any]:
    """清空指定供应商的 API Key（写盘后回读校验）。"""
    with _FILE_LOCK:
        cfg = load_config(force=True)
        cfg.setdefault("api", {}).setdefault(provider, {})
        cfg["api"][provider]["api_key"] = ""
        cfg["api"][provider]["updated_at"] = now_str()
        fresh = save_config(cfg)
    logger.info("[密钥写入] 已清空 provider=%s 的 API Key", provider)
    return fresh


def mask_api_key(plain: str) -> str:
    """将密钥打码用于界面回显，例如 sk-1***abcd。"""
    if not plain:
        return "（未配置）"
    if len(plain) <= 8:
        return plain[0] + "*" * (len(plain) - 1)
    return f"{plain[:4]}****{plain[-4:]}（长度 {len(plain)}）"


def api_key_status(config: Optional[Dict[str, Any]] = None, force: bool = True) -> Dict[str, str]:
    """
    返回两个供应商密钥的脱敏状态，供管理员界面展示。

    【修复点 8｜禁止模式 2】默认 force=True：每次调用都读磁盘最新配置，
    保证管理员保存密钥后，界面刷新立刻能反映真实状态（不会显示“未配置”）。
    """
    cfg = load_config(force=True) if force else (config or load_config(force=False))
    result: Dict[str, str] = {}
    for provider in ("deepseek", "gml"):
        key = get_api_key(provider, cfg, force=False)
        result[provider] = mask_api_key(key) if key else "（未配置）"
    return result


def api_key_status_detail(force: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    返回密钥详细状态（掩码、长度、更新时间、是否已配置），供 UI 更精确地回显。
    同样以磁盘配置为唯一数据源。
    """
    cfg = load_config(force=True) if force else load_config(force=False)
    detail: Dict[str, Dict[str, Any]] = {}
    for provider in ("deepseek", "gml"):
        node = (cfg.get("api", {}) or {}).get(provider, {}) or {}
        key = get_api_key(provider, cfg, force=False)
        detail[provider] = {
            "configured": bool(key),
            "mask": mask_api_key(key) if key else "（未配置）",
            "length": len(key),
            "updated_at": str(node.get("updated_at", "") or ""),
            "model": str(node.get("model", "") or ""),
        }
    return detail


# ---------------------------------------------------------------------------
# 访问日志（Database/access_log.jsonl）
# ---------------------------------------------------------------------------
def append_access_log(ip: str, question: str, role: str = "guest") -> None:
    """
    追加一条访问日志（仅记录提问行为，不记录 LLM 回答内容）。
    每行一个 JSON 对象，异常仅记录日志不影响主流程。
    """
    record = {
        "ip": ip or "unknown",
        "role": role,
        "question": (question or "").strip(),
        "time": now_str(),
        "ts": datetime.now().timestamp(),
    }
    try:
        with _LOG_LOCK:
            ensure_dirs()
            with open(ACCESS_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("写入访问日志失败: %s", exc)


def read_access_log(limit: int = 5000) -> List[Dict[str, Any]]:
    """读取访问日志，返回最近 limit 条记录（跳过损坏行）。"""
    records: List[Dict[str, Any]] = []
    try:
        if not ACCESS_LOG_PATH.exists():
            return records
        with open(ACCESS_LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except Exception:
                    continue  # 单行损坏直接跳过
    except Exception as exc:
        logger.error("读取访问日志失败: %s", exc)
    return records[-limit:]


def summarize_access_log() -> List[List[str]]:
    """
    聚合访问日志，供「查看当前用户」表格使用。
    字段：访问者 IP、最近访问时间、最近一次提问内容、访问次数
    按最近访问时间倒序排列。
    """
    latest: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for rec in read_access_log():
        ip = str(rec.get("ip", "unknown"))
        item = latest.setdefault(ip, {"ip": ip, "time": "", "question": "", "count": 0, "ts": 0.0})
        item["count"] += 1
        ts = float(rec.get("ts", 0) or 0)
        if ts >= item["ts"]:
            item["ts"] = ts
            item["time"] = str(rec.get("time", ""))
            item["question"] = str(rec.get("question", ""))
    rows = [[v["ip"], v["time"], v["question"][:200], str(v["count"])] for v in latest.values()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def clear_access_log() -> None:
    """清空访问日志（管理员维护用）。"""
    try:
        with _LOG_LOCK:
            ensure_dirs()
            _atomic_write_text(ACCESS_LOG_PATH, "")
    except Exception as exc:
        logger.error("清空访问日志失败: %s", exc)


# ---------------------------------------------------------------------------
# 日志初始化
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """初始化全局日志：同时输出到控制台与 Database/app.log。"""
    ensure_dirs()
    root = logging.getLogger()
    if getattr(root, "_rag_logging_ready", False):
        return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    try:
        fh = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass  # 文件日志不可用时仅保留控制台输出
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    # Windows GBK 控制台下遇到 emoji 等字符时只替换、不抛异常
    try:
        if hasattr(sh.stream, "reconfigure"):
            sh.stream.reconfigure(errors="replace")   # type: ignore[attr-defined]
    except Exception:
        pass
    root.addHandler(sh)
    root._rag_logging_ready = True  # type: ignore[attr-defined]
