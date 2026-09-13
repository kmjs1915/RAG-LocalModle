# -*- coding: utf-8 -*-
"""
文件名：main.py
功能：FastAPI 后端入口（替代原 Gradio 版 app.py）
      - 对外提供 REST 接口，前端为 frontend/ 下的纯静态 HTML/CSS/JS
      - JWT 鉴权（PyJWT，HS256）：管理员 / 游客双角色，替代原 gr.State 会话
      - 全部业务逻辑复用 lib/ 目录（本文件不含业务细节，仅做参数校验与编排）

启动：python main.py            （监听 0.0.0.0:4060，访问 http://127.0.0.1:4060）
      python main.py --host 127.0.0.1   （仅本机可访问）

安全说明：本服务仅面向局域网，HTTP 为明文传输，公共网络存在被抓包风险；
         JWT 密钥派生自 Configs/.secret.key，代码中不存在任何硬编码密钥。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lib import chroma_kb, file_loader, llm_client, score_eval, security

logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# 常量与路径（全部相对路径，适配 Windows）
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
CONFIG_PATH = ROOT_DIR / "Configs" / "config.json"
SECRET_KEY_PATH = ROOT_DIR / "Configs" / ".secret.key"

ROLE_ADMIN = "admin"
ROLE_GUEST = "guest"
ALLOWED_ROLES = (ROLE_ADMIN, ROLE_GUEST)

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "local-private-rag"
DEFAULT_TOKEN_HOURS = 12          # 默认有效期，可在 Configs/config.json 的 jwt.expire_hours 覆盖
MIN_TOKEN_HOURS, MAX_TOKEN_HOURS = 1, 72
DENY_ADMIN = "权限不足：该功能仅限管理员使用"

# 服务监听（端口固定常量，命令行 --port 可临时覆盖）
SERVER_PORT = 4060
DEFAULT_HOST = "0.0.0.0"
HTML_MEDIA_TYPE = "text/html; charset=utf-8"

# 请求体长度上限
MAX_QUESTION_CHARS = 10000
MAX_ACCOUNT_CHARS = 200
MAX_API_KEY_CHARS = 512

# 上传中转与响应体常量
SPOOL_DIR_NAME = "rag_upload_spool"
SPOOL_FALLBACK_DIR = ".upload_tmp"
COPY_CHUNK_BYTES = 1024 * 1024
PREVIEW_CHARS = 120                # 调试信息里的片段摘要长度
CONFIDENCE_DECIMALS = 4            # confidence_avg 保留小数位

# 知识库长任务文案
KB_TASK_PENDING = "准备中…"
KB_TASK_BUSY = "已有知识库任务正在执行，请稍候"

# 知识库长任务的进度状态（供前端轮询展示进度）
_KB_TASK: Dict[str, Any] = {
    "running": False, "action": "", "ratio": 0.0, "message": "",
    "started_at": 0.0, "finished_at": 0.0, "ok": None, "detail": "",
}
_KB_TASK_LOCK = threading.Lock()

# 进程内单例
_KB: Optional[chroma_kb.ChromaKB] = None
_KB_LOCK = threading.Lock()

app = FastAPI(
    title="Local-Private-RAG",
    description="本地局域网 RAG 向量检索知识库（FastAPI + 纯静态前端）",
    version="2.0.0",
    docs_url="/api/docs",          # Swagger 便于调试
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

bearer_scheme = HTTPBearer(auto_error=False)


# ===========================================================================
# 基础工具
# ===========================================================================
def get_kb() -> chroma_kb.ChromaKB:
    """获取进程内唯一的 Chroma 知识库实例（配置始终以磁盘为准）。"""
    global _KB
    with _KB_LOCK:
        if _KB is None:
            _KB = chroma_kb.get_kb(security.load_config())
        return _KB


def client_ip(request: Request) -> str:
    """获取访问者 IP（兼容反向代理的 X-Forwarded-For）。"""
    try:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
    except Exception:
        pass
    return "unknown"


def mask_token(token: str) -> str:
    """日志中使用的 token 掩码（绝不打印完整 token）。"""
    return f"{token[:8]}...{token[-6:]}(len={len(token)})" if token else "(空)"


def lan_ip() -> str:
    """获取本机局域网 IP（用于启动提示），失败时回退 127.0.0.1。"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ===========================================================================
# JWT 鉴权
# ===========================================================================
def jwt_secret() -> bytes:
    """
    JWT 签名密钥：派生自 Configs/.secret.key（首次运行自动生成随机主密钥）。
    不硬编码、不入库、不写进配置明文。
    """
    try:
        if SECRET_KEY_PATH.exists():
            raw = SECRET_KEY_PATH.read_bytes()
        else:
            SECRET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
            raw = os.urandom(32)
            with open(SECRET_KEY_PATH, "wb") as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
    except OSError as exc:
        logger.error("读取/生成 JWT 密钥失败，改用进程内临时密钥：%s", exc)
        raw = os.urandom(32)
    return hashlib.sha256(b"local-private-rag|jwt-v1|" + raw).digest()


def token_ttl_hours() -> int:
    """token 有效期（小时），可由 config.json 的 jwt.expire_hours 覆盖。"""
    try:
        node = security.load_config(force=False).get("jwt", {}) or {}
        return max(MIN_TOKEN_HOURS, min(MAX_TOKEN_HOURS, int(node.get("expire_hours", DEFAULT_TOKEN_HOURS))))
    except Exception:
        return DEFAULT_TOKEN_HOURS


def create_token(subject: str, role: str) -> Dict[str, Any]:
    """签发 JWT：包含 sub(账号)、role(角色)、sid(会话标识)、iat、exp。"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(hours=token_ttl_hours())
    payload = {
        "sub": subject,
        "role": role if role in ALLOWED_ROLES else ROLE_GUEST,
        "sid": uuid.uuid4().hex,
        "iss": JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    token = jwt.encode(payload, jwt_secret(), algorithm=JWT_ALGORITHM)
    logger.info("已签发 token：role=%s sub=%s %s", payload["role"], subject, mask_token(token))
    return {
        "token": token,
        "token_type": "Bearer",
        "role": payload["role"],
        "username": subject,
        "expires_at": expire.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "expires_in": int((expire - now).total_seconds()),
    }


def decode_token(token: str) -> Dict[str, Any]:
    """校验 JWT，失败抛 401（过期与非法分别给出可读提示）。"""
    try:
        return jwt.decode(token, jwt_secret(), algorithms=[JWT_ALGORITHM],
                          issuer=JWT_ISSUER, options={"require": ["exp", "sub", "role"]})
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError as exc:
        logger.warning("token 校验失败：%s", exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录状态无效，请重新登录")


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> Dict[str, Any]:
    """
    解析 Bearer token 得到当前用户（游客/管理员均可通过）。
    无 token / token 非法 / token 过期 → 401，前端捕获后自动跳回登录页。
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录或缺少访问令牌")
    payload = decode_token(credentials.credentials)
    role = str(payload.get("role", ROLE_GUEST))
    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录状态无效，请重新登录")
    return {"username": str(payload.get("sub", "")), "role": role, "sid": payload.get("sid", ""),
            "exp": payload.get("exp", 0)}


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    管理员依赖：所有 /api/admin/* 接口都必须挂载本依赖。
    非管理员一律 403（后端强制校验，前端隐藏入口只是体验优化）。
    """
    if user.get("role") != ROLE_ADMIN:
        logger.warning("拒绝越权访问：role=%s user=%s", user.get("role"), user.get("username"))
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=DENY_ADMIN)
    return user


# ===========================================================================
# 请求模型
# ===========================================================================
class LoginRequest(BaseModel):
    username: str = Field(default="", max_length=MAX_ACCOUNT_CHARS)
    password: str = Field(default="", max_length=MAX_ACCOUNT_CHARS)


class ChatRequest(BaseModel):
    question: str = Field(default="", max_length=MAX_QUESTION_CHARS)


class ApiKeyRequest(BaseModel):
    provider: str = Field(default="deepseek")
    api_key: str = Field(default="", max_length=MAX_API_KEY_CHARS)
    set_active: bool = True


class ApiKeyTestRequest(BaseModel):
    provider: str = Field(default="deepseek")
    api_key: str = Field(default="", max_length=MAX_API_KEY_CHARS)


class PasswordRequest(BaseModel):
    old_password: str = Field(default="", max_length=MAX_ACCOUNT_CHARS)
    new_password: str = Field(default="", max_length=MAX_ACCOUNT_CHARS)
    confirm_password: str = Field(default="", max_length=MAX_ACCOUNT_CHARS)


class PersonaRequest(BaseModel):
    content: str = Field(default="")


# ===========================================================================
# 静态资源与页面
# ===========================================================================
def _page(filename: str) -> FileResponse:
    return FileResponse(FRONTEND_DIR / filename, media_type=HTML_MEDIA_TYPE)


@app.get("/", include_in_schema=False)
def page_index() -> FileResponse:
    """模式选择页（登录 / 游客进入）。"""
    return _page("index.html")


@app.get("/index.html", include_in_schema=False)
def page_index_alias() -> FileResponse:
    return _page("index.html")


@app.get("/chat.html", include_in_schema=False)
def page_chat() -> FileResponse:
    """主界面（左侧侧边栏 + 聊天区）。"""
    return _page("chat.html")


# 前端静态资源目录（VS Code 直接编辑 frontend/ 即可）
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    # 204 无响应体：用 Response 而非带 content 的 JSONResponse，避免 h11 报 Content-Length 不匹配
    return Response(status_code=204)


# ===========================================================================
# 鉴权接口
# ===========================================================================
@app.post("/api/login")
def api_login(payload: LoginRequest) -> Dict[str, Any]:
    """
    管理员账号密码登录：bcrypt 哈希校验（lib.security.verify_admin），
    成功返回 JWT token + role。
    """
    ok, msg = security.verify_admin(payload.username, payload.password)
    if not ok:
        # 账号或密码错误统一提示，不泄露账号是否存在
        logger.warning("管理员登录失败：username=%s", payload.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)
    info = create_token(payload.username.strip(), ROLE_ADMIN)
    info["message"] = "登录成功"
    return info


@app.post("/api/guest/login")
def api_guest_login() -> Dict[str, Any]:
    """游客免密登录：直接签发 guest token（仅对话权限）。"""
    info = create_token("guest", ROLE_GUEST)
    info["message"] = "已以普通游客身份进入"
    return info


@app.post("/api/logout")
def api_logout(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    退出登录：JWT 为无状态令牌，服务端不维护会话，前端清除本地 token 即可。
    该接口仅用于记录日志与前端语义完整。
    """
    logger.info("用户退出登录：role=%s user=%s", user.get("role"), user.get("username"))
    return {"ok": True, "message": "已退出登录"}


@app.get("/api/me")
def api_me(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """返回当前登录身份，供前端决定是否渲染管理员侧边栏（游客不渲染 DOM）。"""
    role = user.get("role")
    return {
        "username": user.get("username"),
        "role": role,
        "role_label": "管理员" if role == ROLE_ADMIN else "普通游客",
        "is_admin": role == ROLE_ADMIN,
        "permissions": ["chat"] + (["admin"] if role == ROLE_ADMIN else []),
    }


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    """健康检查（无需鉴权）。"""
    return {"status": "ok", "time": security.now_str()}


# ===========================================================================
# 对话问答（游客与管理员共有）
# ===========================================================================
@app.post("/api/rag/chat")
def api_rag_chat(payload: ChatRequest, request: Request,
                 user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    RAG 问答：清洗问题 → 余弦召回 Top10 → Reranker 重排 Top3 → 置信度 → LLM。
    返回 answer / confidence / tips 等字段，前端把置信度与低置信提示单独成行渲染。
    同时按业务规则记录访问日志（仅记录提问行为，不记录 LLM 回答）。
    """
    question = score_eval.clean_question(payload.question or "")
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请输入问题后再发送")

    ip = client_ip(request)
    security.append_access_log(ip, question, role=user.get("role", ROLE_GUEST))

    persona, _greeting = file_loader.load_persona()
    try:
        result = score_eval.answer_question(question, get_kb(), persona=persona, config=None)
    except Exception as exc:                       # 兜底：任何异常都转成可读提示
        logger.exception("问答流程异常")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"处理问题时出现异常：{exc}")

    # 正文：优先使用模型原始回答；空召回时使用兜底文案
    body = (result.llm_answer or "").strip() or result.answer
    tips = score_eval.LOW_CONFIDENCE_TEXT if (result.low_confidence and not result.no_result) else ""

    return {
        "question": result.question,
        "answer": body,
        "confidence": result.confidence,                 # 0~100
        "confidence_text": f"参考资料置信度：{result.confidence:.1f}%",
        "low_confidence": result.low_confidence,
        "tips": tips,                                    # 低置信时的追加提示文案
        "no_result": result.no_result,
        "error": result.error or "",
        "elapsed_ms": result.elapsed_ms,
        "provider": result.provider,
        "rerank_ok": result.rerank_ok,
        # 调试信息：召回 Top10 与 Reranker 分数（前端折叠展示）
        "debug": {
            "recalled": [
                {
                    "rerank_rank": c.get("rerank_rank"),
                    "recall_rank": c.get("recall_rank"),
                    "rerank_score": c.get("rerank_score"),
                    "similarity": c.get("similarity"),
                    "source": c.get("source"),
                    "chunk_index": c.get("chunk_index"),
                    "preview": (c.get("text") or "")[:PREVIEW_CHARS],
                }
                for c in result.recalled
            ],
            "used": [{"source": c.get("source"), "chunk_index": c.get("chunk_index")} for c in result.used],
            "confidence_avg": round(result.confidence_avg, CONFIDENCE_DECIMALS),
        },
    }


# ===========================================================================
# 管理员功能 1：上传文件
# ===========================================================================
def _spool_dir() -> Path:
    """上传中转目录：优先系统临时目录，不可用时退回项目内 .upload_tmp。"""
    for d in (Path(tempfile.gettempdir()) / SPOOL_DIR_NAME, ROOT_DIR / SPOOL_FALLBACK_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / f".w_{uuid.uuid4().hex}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return d
        except OSError:
            continue
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="无法创建上传中转目录")


@app.post("/api/admin/upload")
def api_admin_upload(files: List[UploadFile] = File(default=[]),
                     user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """
    上传文件到 user_docs（仅 .pdf/.txt/.docx；重名自动 xxx(1).后缀，不覆盖）。
    上传只落盘原始文档，不自动更新向量库 —— 需管理员再执行「重建知识库」。
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请选择要上传的文件")

    spool = _spool_dir()
    staged: List[Path] = []          # 中转文件（保持原始文件名，便于 lib 取到正确文件名）
    tmp_dirs: List[Path] = []
    skipped: List[str] = []
    try:
        for uf in files:
            name = Path(uf.filename or "").name
            if not name:
                skipped.append("（未提供文件名的上传项）")
                continue
            ext = Path(name).suffix.lower()
            if ext not in file_loader.SUPPORTED_EXTS:
                # 非白名单文件仍登记，由 lib 统一给出「仅允许 …」的提示，这里只留一条日志
                logger.warning("上传文件类型不在白名单：%s", name)
            work = spool / uuid.uuid4().hex
            work.mkdir(parents=True, exist_ok=True)
            tmp_dirs.append(work)
            target = work / name        # 保持原名，save_uploaded_files 用文件名作为落盘名
            try:
                # 【修复点】二进制写入；并先把上传流定位到开头，避免上游读过导致落盘 0 字节
                try:
                    uf.file.seek(0)
                except Exception:
                    pass
                with open(target, "wb") as fh:
                    shutil.copyfileobj(uf.file, fh, COPY_CHUNK_BYTES)
                    fh.flush()
                    os.fsync(fh.fileno())
                written = target.stat().st_size
                expected = getattr(uf, "size", None)
                if written <= 0:
                    skipped.append(f"{name}：上传内容为空（0 字节）")
                    logger.error("上传内容为空：%s", name)
                    continue
                if expected and written != expected:
                    logger.warning("上传大小与声明不一致：%s（声明 %s 字节，实际 %d 字节）", name, expected, written)
                logger.info("上传中转完成：%s（%d 字节，类型 %s）", name, written, ext or "无扩展名")
            except PermissionError as exc:
                skipped.append(f"{name}：中转文件被占用或无写入权限（{exc}）")
                logger.error("上传中转被拒绝 %s -> %s", name, exc)
                continue
            except OSError as exc:
                skipped.append(f"{name}：中转写入失败（{type(exc).__name__}: {exc}）")
                logger.error("上传中转写入失败 %s -> %s", name, exc)
                continue
            except Exception as exc:                     # 兜底：任何异常都要给出可读原因
                skipped.append(f"{name}：上传处理异常（{type(exc).__name__}: {exc}）")
                logger.exception("上传处理出现未预期异常：%s", name)
                continue
            staged.append(target)

        saved, lib_skipped = file_loader.save_uploaded_files(staged)
        skipped.extend(lib_skipped)
    finally:
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    logger.info("管理员上传：保存 %d 个，跳过 %d 个", len(saved), len(skipped))
    return {
        "ok": bool(saved),
        "saved": saved,
        "skipped": skipped,
        "message": (f"已保存 {len(saved)} 个文件到 user_docs" if saved else "没有文件被保存"),
        "tip": "上传后向量库不会自动更新，请执行「重建知识库」后新文件才会生效。",
    }


# ===========================================================================
# 管理员功能 2：访问日志
# ===========================================================================
@app.get("/api/admin/logs")
def api_admin_logs(user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """访问日志：访问者 IP / 最近访问时间 / 最近一次提问 / 访问次数（仅记录提问行为）。"""
    rows = security.summarize_access_log()
    return {
        "ok": True,
        "total_users": len(rows),
        "total_records": len(security.read_access_log()),
        "rows": [{"ip": r[0], "time": r[1], "question": r[2], "count": r[3]} for r in rows],
    }


# ===========================================================================
# 管理员功能 3：API 密钥管理
# ===========================================================================
@app.get("/api/admin/apikey")
def api_admin_apikey_get(user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """读取两个供应商的密钥状态（掩码 + 长度 + 更新时间），始终读磁盘最新配置。"""
    detail = security.api_key_status_detail()

    def provider_info(provider: str) -> Dict[str, Any]:
        info = detail.get(provider) or {}
        return {
            "provider": provider,
            "label": llm_client.PROVIDER_LABELS.get(provider, provider),
            "model": info.get("model") or llm_client.get_provider_config(provider).get("model", ""),
            "configured": bool(info.get("configured", False)),
            "mask": info.get("mask", "（未配置）"),
            "length": info.get("length", 0),
            "updated_at": info.get("updated_at", ""),
        }

    return {
        "ok": True,
        "active_provider": llm_client.get_active_provider(),
        "providers": [provider_info(p) for p in llm_client.PROVIDERS],
    }


@app.post("/api/admin/apikey/test")
def api_admin_apikey_test(payload: ApiKeyTestRequest,
                          user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """
    连通测试：
      - 传了 api_key → 测试输入框里的临时密钥（保存前预检）
      - 未传 api_key → 读取磁盘配置里的密钥实测（保存后的真实连通状态）
    """
    provider = (payload.provider or "").strip().lower()
    if provider not in llm_client.PROVIDERS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的供应商：{provider}")
    if (payload.api_key or "").strip():
        ok, msg = llm_client.test_connection(provider, api_key=payload.api_key.strip())
        source = "临时输入密钥"
    else:
        ok, msg = llm_client.test_saved_connection(provider)
        source = "磁盘配置"
    return {"ok": ok, "provider": provider, "source": source, "message": msg}


@app.post("/api/admin/apikey")
def api_admin_apikey_save(payload: ApiKeyRequest,
                          user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """
    保存 API 密钥（两段式校验）：临时密钥连通测试 → 写盘 → 回读磁盘二次校验。
    任一步失败即拒绝保存（二次校验失败自动回滚），失败返回 400 与原因。
    """
    provider = (payload.provider or "").strip().lower()
    if provider not in llm_client.PROVIDERS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的供应商：{provider}")
    if not (payload.api_key or "").strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请填写新的 API Key")

    result = llm_client.save_and_verify_api_key(provider, payload.api_key.strip(),
                                                set_active=bool(payload.set_active))
    result.pop("config", None)                     # 不回传完整配置（含密文），减小响应体
    if not result.get("ok"):
        logger.warning("API Key 保存失败：provider=%s stage=%s", provider, result.get("stage"))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("message", "保存失败"))
    return result


# ===========================================================================
# 管理员功能 4：打开原始文档目录
# ===========================================================================
@app.post("/api/admin/open-docs")
def api_admin_open_docs(user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """调用系统资源管理器打开 user_docs（仅浏览，不做文件操作）。"""
    ok, msg = file_loader.open_user_docs_folder()
    return {"ok": ok, "message": msg,
            "tip": "手动增删文件后，必须执行「重建知识库」才会同步到向量库。"}


# ===========================================================================
# 管理员功能 5：修改密码
# ===========================================================================
@app.post("/api/admin/password")
def api_admin_password(payload: PasswordRequest,
                       user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """修改管理员密码：旧密码 bcrypt 校验 + 两次新密码一致 + 强度校验。"""
    ok, msg = security.change_admin_password(payload.old_password, payload.new_password,
                                             payload.confirm_password)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return {"ok": True, "message": msg}


# ===========================================================================
# 管理员功能 6：Agent 人设
# ===========================================================================
@app.get("/api/admin/persona")
def api_admin_persona_get(user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """读取 agent_config/Person.txt 内容（网页内编辑，不唤起本地记事本）。"""
    text = file_loader.load_persona_text()
    return {"ok": True, "content": text, "length": len(text),
            "updated_at": file_loader.persona_updated_at()}


@app.post("/api/admin/persona")
def api_admin_persona_save(payload: PersonaRequest,
                           user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """保存 Agent 人设（覆盖 Person.txt），下次提问即生效。"""
    ok, msg = file_loader.save_persona_text(payload.content)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return {"ok": True, "message": msg, "length": len(payload.content or "")}


# ===========================================================================
# 管理员功能 7：重建知识库（写操作由 lib 加文件锁）
# ===========================================================================
def _claim_kb_task(action: str) -> None:
    """
    原子地抢占知识库长任务（检查 + 置位在同一把锁内完成，避免并发双写）；
    已有任务在跑时抛 409。
    """
    with _KB_TASK_LOCK:
        if _KB_TASK["running"]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=KB_TASK_BUSY)
        _KB_TASK.update(running=True, action=action, ratio=0.0, message=KB_TASK_PENDING,
                        started_at=time.time(), finished_at=0.0, ok=None, detail="")


def _finish_kb_task(ok: bool, detail: str = "", ratio: Optional[float] = None,
                    message: Optional[str] = None) -> None:
    """结束知识库长任务（可选：把最终进度/文案一并落定，供前端最后一次轮询读取）。"""
    with _KB_TASK_LOCK:
        if ratio is not None:
            _KB_TASK["ratio"] = max(0.0, min(1.0, float(ratio)))
        if message is not None:
            _KB_TASK["message"] = message
        _KB_TASK.update(running=False, finished_at=time.time(), ok=ok, detail=detail)


def _kb_progress(ratio: float, desc: str = "") -> None:
    """lib 的进度回调（0~1 + 描述），写入共享状态供前端轮询。"""
    with _KB_TASK_LOCK:
        _KB_TASK["ratio"] = max(0.0, min(1.0, float(ratio)))
        _KB_TASK["message"] = desc or ""


# ---------------------------------------------------------------------------
# 重建知识库后的 Chroma 残留目录清理（仅 Windows）
# ---------------------------------------------------------------------------
CHROMA_SQLITE_NAME = "chroma.sqlite3"      # Chroma 元数据库文件名，清理过程中绝不会被删除
# Chroma HNSW segment 目录内的已知文件（含未知文件时保守跳过，避免误删用户数据）
CHROMA_SEGMENT_FILES = ("data_level0.bin", "header.bin", "length.bin",
                        "link_lists.bin", "index_metadata.pickle")
# 严格 UUID 命名（Chroma 的 segment / collection 目录均为此格式）
_UUID_DIR_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                          r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _chroma_in_use_ids(sqlite_path: Path) -> Optional[set]:
    """
    只读查询 chroma.sqlite3，返回仍被 Chroma 引用的 id 集合（segments.id ∪ collections.id）。

    - 使用 mode=ro + PRAGMA query_only 打开，绝不修改/删除该 sqlite 库；
    - 返回 None 表示无法判定引用关系（库不存在/被占用/表结构异常），
      调用方必须放弃本次清理，避免把正在使用的目录误删。
    """
    if not sqlite_path.is_file():
        logger.info("[残留清理] 未找到 %s，跳过清理", CHROMA_SQLITE_NAME)
        return None
    try:
        con = sqlite3.connect(f"file:/{sqlite_path.as_posix()}?mode=ro", uri=True, timeout=5)
    except Exception as exc:
        logger.warning("[残留清理] 只读打开 %s 失败：%s", CHROMA_SQLITE_NAME, exc)
        return None
    try:
        con.execute("PRAGMA query_only = ON;")
        try:
            ids = {str(v).strip().lower() for (v,) in con.execute("SELECT id FROM segments") if v}
        except Exception as exc:
            logger.warning("[残留清理] 读取 segments 表失败（%s），放弃本次清理以避免误删", exc)
            return None
        try:
            # 兼容「按 collection id 命名目录」的 Chroma 版本：一并纳入白名单
            ids |= {str(v).strip().lower() for (v,) in con.execute("SELECT id FROM collections") if v}
        except Exception as exc:
            logger.warning("[残留清理] 读取 collections 表失败（忽略，不影响判断）：%s", exc)
        return ids
    except Exception as exc:
        logger.warning("[残留清理] 查询元数据失败：%s", exc)
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass


def _looks_like_segment_dir(path: Path) -> bool:
    """
    判断是否为 Chroma 的 HNSW segment 目录：空目录，或只含已知的 segment 文件。
    含子目录或陌生文件时返回 False（保守跳过，宁可留残留也不误删）。
    """
    try:
        entries = list(path.iterdir())
    except Exception:
        return False
    if not entries:
        return True
    for item in entries:
        if item.is_dir():                                   # 含子目录 -> 不是标准 segment 目录
            return False
        if item.name not in CHROMA_SEGMENT_FILES and not item.name.endswith((".bin", ".pickle")):
            return False                                    # 含陌生文件 -> 保守跳过
    return True


def cleanup_orphan_chroma_dirs() -> Dict[str, Any]:
    """
    Windows 专用：删除 chroma_db 下「未被 chroma.sqlite3 引用」的孤立 UUID 残留目录。

    背景：Chroma 删除集合时其 HNSW 索引目录（UUID 命名）在 Windows 上可能因文件被占用而
    残留，反复重建会不断堆积。本函数在【重建知识库成功后】调用，安全约束如下：
      - 仅在 Windows 平台执行，其他平台直接跳过；
      - 只遍历 chroma_db 的直接子目录，且只处理严格 UUID 命名的文件夹；
      - 跳过 chroma.sqlite3 等一切文件、符号链接、非 UUID 目录、含未知内容的 UUID 目录；
      - 跳过仍被 sqlite 元数据引用（segments.id / collections.id）的目录；
      - 删除前再次校验目标确实位于 chroma_db 之内；
      - 任何异常只写日志，绝不影响重建知识库主流程。
    返回统计字典（supported/referenced/orphan/removed/failed）。
    """
    summary: Dict[str, Any] = {"supported": False, "referenced": 0, "orphan": 0,
                               "removed": 0, "failed": 0}
    if not sys.platform.startswith("win"):                  # 仅 Windows 执行，Linux 跳过
        logger.info("[残留清理] 当前平台为 %s，跳过 Chroma 残留目录清理", sys.platform)
        return summary
    summary["supported"] = True
    try:
        chroma_dir = Path(get_kb().persist_dir).resolve()
    except Exception as exc:
        logger.warning("[残留清理] 获取 Chroma 目录失败：%s", exc)
        return summary
    if not chroma_dir.is_dir():
        logger.warning("[残留清理] Chroma 目录不存在：%s", chroma_dir)
        return summary

    in_use = _chroma_in_use_ids(chroma_dir / CHROMA_SQLITE_NAME)
    if in_use is None:
        return summary                                     # 无法判定引用关系 -> 不删任何东西
    summary["referenced"] = len(in_use)

    chroma_real = Path(os.path.realpath(chroma_dir))
    orphan_dirs: List[Path] = []
    try:
        entries = list(chroma_dir.iterdir())               # 只遍历 chroma_db 的直接子项
    except Exception as exc:
        logger.warning("[残留清理] 遍历 %s 失败：%s", chroma_dir, exc)
        return summary

    for entry in entries:
        try:
            if entry.is_symlink() or not entry.is_dir():
                continue                                   # 跳过文件（含 chroma.sqlite3）与链接
            if not _UUID_DIR_RE.match(entry.name):
                continue                                   # 非 UUID 目录一律不动
            if entry.name.lower() in in_use:
                continue                                   # 仍被元数据引用 -> 正在使用，保留
            if Path(os.path.realpath(entry)).parent != chroma_real:
                continue                                   # 双重校验：必须位于 chroma_db 内
            if not _looks_like_segment_dir(entry):
                logger.info("[残留清理] %s 含未知内容，保守跳过", entry.name)
                continue
            orphan_dirs.append(entry)
        except Exception as exc:
            logger.warning("[残留清理] 检查 %s 失败（跳过该项）：%s", entry, exc)
    summary["orphan"] = len(orphan_dirs)

    for path in orphan_dirs:
        for attempt in (1, 2):                             # 文件可能被短暂占用，重试一次
            try:
                shutil.rmtree(path)
                summary["removed"] += 1
                logger.info("[残留清理] 已删除孤立索引目录：%s", path.name)
                break
            except Exception as exc:
                if attempt == 2:
                    summary["failed"] += 1
                    logger.warning("[残留清理] 删除失败（仅记录日志，不影响重建结果）：%s -> %s",
                                   path.name, exc)
                else:
                    time.sleep(0.3)
    logger.info("[残留清理] 结束：元数据引用 %d 个，孤立 %d 个，已删除 %d 个，失败 %d 个（%s 已保留）",
                summary["referenced"], summary["orphan"], summary["removed"],
                summary["failed"], CHROMA_SQLITE_NAME)
    return summary


@app.post("/api/admin/kb/reinit")
def api_admin_kb_reinit(user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """
    重建知识库：清空 personal_kb → 重新扫描 user_docs 全量切片入库。
    lib 内部已加文件锁，防止并发写坏 Chroma。
    """
    _claim_kb_task("reinit")
    try:
        result = get_kb().reset(progress=_kb_progress)
        detail = result.get("summary", "")
        if result.get("errors"):
            detail += f"；跳过 {len(result['errors'])} 个文件"
        logger.info("知识库重建完成：%s", detail)

        # 【本次改动】重建成功后清理 Windows 下的 Chroma 残留索引目录（孤立 UUID 文件夹）。
        # 只在重建成功之后执行；清理异常只记日志，绝不中断重建结果。
        cleanup_note = ""
        try:
            cleanup = cleanup_orphan_chroma_dirs()
            if cleanup.get("removed"):
                cleanup_note += f"（已清理 {cleanup['removed']} 个残留索引目录）"
            if cleanup.get("failed"):
                cleanup_note += f"（{cleanup['failed']} 个残留目录删除失败，详见日志）"
        except Exception as exc:
            logger.warning("[残留清理] 执行异常，已忽略：%s", exc)

        _finish_kb_task(True, detail)
        return {
            "ok": True,
            "chunk_count": result.get("chunk_count", 0),
            "files_total": result.get("files_total", 0),
            "files_ok": result.get("files_ok", 0),
            "errors": result.get("errors", []),
            "message": f"知识库重建完成：{detail}{cleanup_note}",
        }
    except Exception as exc:
        logger.error("知识库重建失败：%s", exc)
        _finish_kb_task(False, str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"重建失败：{exc}")


@app.get("/api/admin/kb/progress")
def api_admin_kb_progress(user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """知识库长任务进度（前端在重建知识库时轮询展示）。"""
    with _KB_TASK_LOCK:
        task = dict(_KB_TASK)
    if task["running"]:
        task["elapsed"] = round(time.time() - task["started_at"], 1)
    else:
        task["elapsed"] = round(max(0.0, task["finished_at"] - task["started_at"]), 1)
    return task


# ===========================================================================
# 状态查询（顶部栏 / 侧边栏展示）
# ===========================================================================
@app.get("/api/status")
def api_status(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """知识库与模型状态概览（对话页顶部栏、侧边栏使用；同时返回开场白问候语）。"""
    try:
        stats = get_kb().stats()
    except Exception as exc:
        logger.error("读取知识库状态失败：%s", exc)
        stats = {"collection": "-", "chunk_count": 0, "file_count": 0, "files": []}
    docs = file_loader.scan_user_docs()
    # 只取人设文件中的「开场白」一行，不向前端暴露完整人设（避免泄露系统提示词）
    _persona, greeting = file_loader.load_persona()
    return {
        "ok": True,
        "kb": {
            "collection": stats.get("collection"),
            "chunk_count": stats.get("chunk_count", 0),
            "file_count": stats.get("file_count", 0),
            "files": stats.get("files", []),
        },
        "docs_count": len(docs),
        "docs_pending": max(0, len(docs) - int(stats.get("file_count", 0) or 0)),
        "active_provider": llm_client.get_active_provider(),
        "welcome": greeting,
        "role": user.get("role"),
    }


# ===========================================================================
# 统一异常返回（保证前端拿到的永远是 JSON，便于提示）
# ===========================================================================
@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code,
                        content={"ok": False, "detail": exc.detail, "status": exc.status_code})


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理异常：%s", exc)
    return JSONResponse(status_code=500,
                        content={"ok": False, "detail": f"服务器内部错误：{exc}", "status": 500})


# ===========================================================================
# 启动
# ===========================================================================
def warmup_models() -> None:
    """后台预热 Embedding / Reranker，避免首个提问等待过久。"""
    for label, warmup in (("Embedding", lambda: get_kb().embedder.warmup()),
                          ("Reranker", lambda: get_kb().reranker.warmup())):
        try:
            warmup()
        except Exception as exc:
            logger.error("%s 预热失败：%s", label, exc)


def startup_self_check() -> List[str]:
    """启动自检：模型文件、向量库、配置、密钥状态。"""
    from lib import embedding as embedding_mod, reranker as reranker_mod

    cfg = security.load_config(force=True)
    lines: List[str] = []

    for label, checker, cfg_key in (("Embedding", embedding_mod.check_embedding_model, "embedding"),
                                    ("Reranker", reranker_mod.check_reranker_model, "reranker")):
        ok, msg = checker(cfg.get(cfg_key, {}).get("model_path"))
        lines.append(("[OK]   " if ok else "[失败] ") + msg)

    security.ensure_security_file()
    lines.append(f"[OK]   安全配置就绪：{security.SECURITY_PATH}")
    if security.is_default_admin_password():
        lines.append("[警告] 管理员仍在使用默认密码，请尽快在「修改密码」中更换")

    try:
        stats = get_kb().stats()
        lines.append(f"[OK]   向量库：{stats.get('persist_dir')}，集合 {stats.get('collection')}，"
                     f"切片 {stats.get('chunk_count', 0)} 条，来源文件 {stats.get('file_count', 0)} 个")
    except Exception as exc:
        lines.append(f"[警告] 向量库读取异常：{exc}")

    lines.append(f"[OK]   user_docs 目录文档数：{len(file_loader.scan_user_docs())}")
    detail = security.api_key_status_detail()
    keys = " ｜ ".join(f"{p}：{(detail.get(p) or {}).get('mask', '（未配置）')}"
                      for p in llm_client.PROVIDERS)
    lines.append(f"[OK]   {keys} ｜ 当前启用：{llm_client.get_active_provider()}")
    return lines


def main() -> None:
    """命令行入口：python main.py [--host 0.0.0.0] [--port 4060] [--reload]"""
    parser = argparse.ArgumentParser(description="Local-Private-RAG FastAPI 服务")
    parser.add_argument("--host", default=None, help="监听地址，默认取 config.json 的 web.host")
    parser.add_argument("--port", type=int, default=None, help=f"监听端口，默认 {SERVER_PORT}")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载")
    args = parser.parse_args()

    security.setup_logging()
    security.ensure_dirs()

    cfg = security.load_config(force=True)
    web_cfg = cfg.get("web", {}) or {}
    server_cfg = cfg.get("server", {}) or {}
    # 监听地址/端口：命令行 > 环境变量 > config.json > 缺省值
    host = args.host or os.environ.get("RAG_HOST") or str(
        web_cfg.get("host") or server_cfg.get("server_name") or DEFAULT_HOST)
    port = args.port or SERVER_PORT

    print("=" * 78)
    print("本地 RAG 向量检索知识库 —— FastAPI 服务启动自检")
    print("=" * 78)
    for line in startup_self_check():
        print(line)
    print("-" * 78)
    print(f"前端页面：http://127.0.0.1:{port}/            （模式选择页）")
    print(f"          http://127.0.0.1:{port}/chat.html   （主界面）")
    print(f"接口文档：http://127.0.0.1:{port}/api/docs")
    if host == DEFAULT_HOST:
        print(f"局域网访问：http://{lan_ip()}:{port}   （如需仅本机访问：python main.py --host 127.0.0.1）")
    print("安全提示：仅面向局域网，HTTP 为明文传输，公共网络存在被抓包风险；")
    print("          JWT 鉴权，token 由浏览器本地存储保存，服务端不维护会话状态。")
    print("=" * 78)

    threading.Thread(target=warmup_models, name="model-warmup", daemon=True).start()
    uvicorn.run("main:app" if args.reload else app, host=host, port=port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
