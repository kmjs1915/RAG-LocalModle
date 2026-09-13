# -*- coding: utf-8 -*-
"""
文件名：lib/file_loader.py
功能：文件解析与文本切片层
      1) PDF(pdfplumber) / TXT(多编码兜底) / DOCX(python-docx) 读取
      2) 文本规范化与滑动窗口切片（chunk_size=800、chunk_overlap=150）
      3) user_docs 目录扫描、上传文件落盘（重名自动 xxx(1).后缀）
      4) Agent 人设文件（agent_config/Person.txt）读写
      5) 打开 user_docs 原始文档目录（仅浏览）
异常策略：单个文件损坏/乱码/为空时，记录日志并跳过，绝不中断整体流程。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lib.security import PERSONA_PATH, ROOT_DIR, USER_DOCS_DIR, ensure_dirs, now_str

logger = logging.getLogger(__name__)

# 允许的文档类型与大小限制
SUPPORTED_EXTS = (".pdf", ".txt", ".docx")
MAX_FILE_MB = 100                     # 单个文件最大 100MB，超过直接跳过
MAX_EXTRACT_CHARS = 2_000_000         # 单文件最大抽取字符数，防止超长文本拖垮内存
PERSONA_MAX_CHARS = 200_000           # 人设文件最大长度

# TXT 编码探测顺序（中文环境常见编码）
TXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "utf-16", "latin-1")

# 句末标点，用于"优先在语义边界断开"
_SENTENCE_END = "。！？!?；;…"


class DocumentLoadError(Exception):
    """文档解析失败（可被上层捕获并跳过）。"""


@dataclass
class Chunk:
    """一个文本切片及其元数据。"""
    text: str                                  # 切片原文（不含前缀，用于展示与拼 prompt）
    source: str                                # 来源文件名
    chunk_index: int                           # 同一文件内的切片序号
    char_start: int = 0                        # 在原文中的起始位置（便于溯源）


@dataclass
class LoadReport:
    """整批文档解析结果汇总。"""
    chunks: List[Chunk] = field(default_factory=list)
    files_total: int = 0
    files_ok: int = 0
    errors: List[str] = field(default_factory=list)   # 形如 "xxx.pdf: 解析失败原因"

    def add_error(self, filename: str, reason: str) -> None:
        msg = f"{filename}: {reason}"
        self.errors.append(msg)
        logger.warning("跳过文件 %s", msg)

    @property
    def summary(self) -> str:
        text = f"扫描文件 {self.files_total} 个，成功 {self.files_ok} 个，切片 {len(self.chunks)} 条"
        if self.errors:
            text += f"，跳过 {len(self.errors)} 个"
        return text


# ---------------------------------------------------------------------------
# 文件读取
# ---------------------------------------------------------------------------
def read_txt(path: Path) -> str:
    """
    读取 TXT：依次尝试多种编码，全部失败则以替换字符兜底（不抛异常中断）。
    【修复点】区分为「文件被占用/无权限」与「编码无法识别」，便于前端提示。
    """
    last_error: Optional[Exception] = None
    for enc in TXT_ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as fh:      # 文本文件按文本模式读取
                return fh.read()
        except PermissionError as exc:
            logger.error("TXT 被占用或无读取权限：%s -> %s", path, exc)
            raise DocumentLoadError("文件被其他程序占用或无读取权限（请关闭后重试）") from exc
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except OSError as exc:  # 磁盘/IO 等异常
            logger.error("TXT 读取 IO 异常：%s -> %s", path, exc)
            raise DocumentLoadError(f"文件无法读取（{type(exc).__name__}: {exc}）") from exc
    try:  # 最后兜底：忽略无法解码的字节
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")
    except PermissionError as exc:
        raise DocumentLoadError("文件被其他程序占用或无读取权限（请关闭后重试）") from exc
    except Exception as exc:
        raise DocumentLoadError(f"编码无法识别：{last_error or exc}") from exc


def _precheck_binary_file(path: Path, signature: bytes, signature_name: str,
                          legacy: Optional[Tuple[bytes, str]] = None) -> None:
    """
    PDF / DOCX 的二进制预检（统一用 rb 二进制模式打开，绝不用文本模式 r）：

      1) 文件被其他程序占用 / 无权限   -> DocumentLoadError（可读的占用提示）
      2) 空文件（0 字节）              -> DocumentLoadError
      3) 命中 legacy 魔数（如 .doc 老格式）-> 给出针对性的转换建议
      4) 魔数不匹配（格式不符/损坏）    -> DocumentLoadError，并记录前 8 字节便于定位
      5) 路径含中文/空格同样适用（Path 已正确处理，不做任何字符串拼接）

    仅做检查，不读取全部内容，因此对大文件没有额外开销。
    """
    try:
        # 二进制只读：既能读魔数，也能正确探测「文件被独占锁定」
        with open(path, "rb") as fh:
            head = fh.read(max(16, len(signature)))
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
    except PermissionError as exc:
        logger.error("文件被占用或无读取权限：%s -> %s", path, exc)
        raise DocumentLoadError("文件被其他程序占用或无读取权限（请关闭后重试）") from exc
    except OSError as exc:
        logger.error("文件读取 IO 异常：%s -> %s", path, exc)
        raise DocumentLoadError(f"文件无法读取（{type(exc).__name__}: {exc}）") from exc

    if size == 0:
        raise DocumentLoadError("文件内容为空（0 字节）")
    if head.startswith(signature):
        return
    if legacy and head.startswith(legacy[0]):
        logger.warning("文件为旧格式：%s（%s）", path, legacy[1])
        raise DocumentLoadError(legacy[1])
    logger.warning("文件格式校验未通过：%s（期望 %s，实际前 8 字节 %r，大小 %d 字节）",
                   path, signature_name, head[:8], size)
    raise DocumentLoadError(
        f"文件格式不正确或已损坏（缺少 {signature_name}，前 8 字节为 {head[:8]!r}）"
    )


def read_pdf(path: Path) -> str:
    """
    读取 PDF：pdfplumber 逐页抽取文本，单页异常跳过该页。

    【修复点】先做二进制预检（rb 打开 + %PDF 魔数 + 空文件/被占用判断），
    再交给 pdfplumber 按路径解析，报错按类型区分（占用 / 损坏 / 加密 / 无文本层）。
    """
    _precheck_binary_file(path, b"%PDF", "PDF 标识 %PDF")
    try:
        import pdfplumber
    except Exception as exc:  # 依赖缺失
        logger.exception("导入 pdfplumber 失败")
        raise DocumentLoadError(f"pdfplumber 不可用（{type(exc).__name__}: {exc}）") from exc

    pages: List[str] = []
    page_errors = 0
    try:
        # pdfplumber.open 支持路径与文件对象；此处传字符串路径，库内部以二进制方式打开
        with pdfplumber.open(str(path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    pages.append(page.extract_text() or "")
                except Exception as exc:
                    page_errors += 1
                    logger.warning("PDF 第 %d 页解析失败 %s: %s", index, path.name, exc)
    except PermissionError as exc:
        logger.error("PDF 被占用或无权读取 %s: %s", path.name, exc)
        raise DocumentLoadError(f"文件被其他程序占用或无读取权限（{exc}）") from exc
    except Exception as exc:
        cls = type(exc).__name__
        logger.exception("PDF 解析异常：%s", path.name)          # 底层报错 + 堆栈写入日志
        if "Password" in cls:
            raise DocumentLoadError("PDF 已加密，需要密码才能解析") from exc
        if "Syntax" in cls or "Pdfminer" in cls or "PSException" in cls:
            raise DocumentLoadError(f"PDF 结构损坏（{cls}: {exc}）") from exc
        raise DocumentLoadError(f"PDF 解析未知异常（{cls}: {exc}）") from exc

    text = "\n".join(pages)
    if not text.strip():
        # 无文本层：最常见是扫描件/图片版 PDF
        raise DocumentLoadError(
            "PDF 内无可用文本层（可能是扫描图片 PDF，需 OCR 处理）"
            + (f"；另有 {page_errors} 页抽取失败" if page_errors else "")
        )
    logger.info("PDF 解析成功：%s，%d 页，%d 字符", path.name, len(pages), len(text))
    return text


def read_docx(path: Path) -> str:
    """
    读取 DOCX：python-docx 抽取段落与表格文本。

    【修复点】导入语句统一为 `from docx import Document`；
    先做二进制预检（rb 打开 + ZIP 魔数 PK，用于区分「.doc 改名」与「文件损坏」）。
    """
    # docx 本质是 ZIP 包（PK\x03\x04）；.doc 老格式是 OLE2，据此给出针对性提示
    _precheck_binary_file(
        path, b"PK\x03\x04", "DOCX(ZIP) 标识 PK",
        legacy=(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
                "该文件是 Word 97-2003 的 .doc 老格式（OLE2），请用 Word 另存为 .docx 后再上传"),
    )
    try:
        from docx import Document                       # 需求1：确认使用该导入方式
    except Exception as exc:
        logger.exception("导入 python-docx 失败")
        raise DocumentLoadError(f"python-docx 不可用（{type(exc).__name__}: {exc}）") from exc

    try:
        document = Document(str(path))
        parts: List[str] = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        # 表格内容一并抽取（表格文字常是知识库关键信息）
        for table in getattr(document, "tables", []):
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
    except PermissionError as exc:
        logger.error("DOCX 被占用或无权读取 %s: %s", path.name, exc)
        raise DocumentLoadError(f"文件被其他程序占用或无读取权限（{exc}）") from exc
    except Exception as exc:
        cls = type(exc).__name__
        logger.exception("DOCX 解析异常：%s", path.name)          # 底层报错 + 堆栈写入日志
        if cls in ("BadZipFile", "PackageNotFoundError", "CRCError"):
            raise DocumentLoadError(
                f"不是有效的 DOCX（可能是 .doc 老格式被改名为 .docx，或文件已损坏；{cls}: {exc}）"
            ) from exc
        raise DocumentLoadError(f"DOCX 解析未知异常（{cls}: {exc}）") from exc

    text = "\n".join(parts)
    if not text.strip():
        raise DocumentLoadError("DOCX 内无可用文本（文档可能是空的，或正文全为图片）")
    logger.info("DOCX 解析成功：%s，%d 段，%d 字符", path.name, len(parts), len(text))
    return text


def load_document(path: Path) -> str:
    """
    按扩展名分发读取，返回原始文本。
    任何失败都抛 DocumentLoadError（由调用方捕获并跳过）。
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise DocumentLoadError("文件不存在")
    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > MAX_FILE_MB:
        raise DocumentLoadError(f"文件过大（{size_mb:.1f}MB > {MAX_FILE_MB}MB）")
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = read_pdf(path)
    elif ext == ".txt":
        text = read_txt(path)
    elif ext == ".docx":
        text = read_docx(path)
    else:
        raise DocumentLoadError(f"不支持的文件类型：{ext or '未知'}")
    if not text or not text.strip():
        # 与 _precheck_binary_file 的空文件提示保持一致，并区分"0 字节"与"只有空白字符"
        try:
            file_size = path.stat().st_size
        except OSError:
            file_size = -1
        if file_size == 0:
            raise DocumentLoadError("文件内容为空（0 字节）")
        logger.warning("文件解析后无有效文本：%s（%d 字节）", path.name, file_size)
        raise DocumentLoadError("文件无有效文本内容（仅空白字符，无法切片）")
    if len(text) > MAX_EXTRACT_CHARS:
        logger.warning("文件 %s 文本过长，已截断至 %s 字符", path.name, MAX_EXTRACT_CHARS)
        text = text[:MAX_EXTRACT_CHARS]
    return text


# ---------------------------------------------------------------------------
# 文本规范化与切片
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """规范化：统一换行、去零宽字符、压缩连续空白（保留段落换行）。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_atoms(text: str) -> List[str]:
    """
    把文本拆成"语义原子"（段落 → 句子），用于切片时优先在句末断开。
    原子拼接后与原文内容一致（不含分隔换行），不会被改写。
    """
    atoms: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        buf = ""
        for ch in para:
            buf += ch
            if ch in _SENTENCE_END:
                atoms.append(buf)
                buf = ""
        if buf.strip():
            atoms.append(buf)
    return atoms


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    滑动窗口切片：
      - chunk_size   目标切片长度（字符），默认 800
      - chunk_overlap 相邻切片重叠长度（字符），默认 150
      - 优先在句子边界断开；超长单句按窗口硬切
    返回切片文本列表（已去重、去空白）。
    """
    chunk_size = max(50, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    text = normalize_text(text)
    if not text:
        return []

    atoms = _split_atoms(text)
    if not atoms:
        return []

    # 1) 超长原子（如无标点的长段落）按窗口硬切
    pieces: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for atom in atoms:
        if len(atom) <= chunk_size:
            pieces.append(atom)
            continue
        for i in range(0, len(atom), step):
            piece = atom[i:i + chunk_size]
            if piece.strip():
                pieces.append(piece)
            if i + chunk_size >= len(atom):
                break

    # 2) 贪心合并，并在切片尾部保留 overlap 长度的内容
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for piece in pieces:
        if cur and cur_len + len(piece) > chunk_size:
            chunks.append("".join(cur))
            keep: List[str] = []
            keep_len = 0
            for prev in reversed(cur):
                if keep_len + len(prev) > chunk_overlap:
                    break
                keep.insert(0, prev)
                keep_len += len(prev)
            cur, cur_len = keep, keep_len
        cur.append(piece)
        cur_len += len(piece)
    if cur:
        chunks.append("".join(cur))

    # 3) 清洗、去重
    result: List[str] = []
    seen = set()
    for c in chunks:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        result.append(c)
    return result


def load_and_split(path: Path, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Chunk]:
    """读取单个文件并切片，返回 Chunk 列表（失败抛 DocumentLoadError）。"""
    text = load_document(path)
    pieces = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not pieces:
        raise DocumentLoadError("切片结果为空")
    return [Chunk(text=p, source=Path(path).name, chunk_index=i) for i, p in enumerate(pieces)]


# ---------------------------------------------------------------------------
# 目录扫描与批量入库准备
# ---------------------------------------------------------------------------
def scan_user_docs(docs_dir: Optional[Path] = None) -> List[Path]:
    """扫描 user_docs 目录下所有 .pdf/.txt/.docx 文件（按文件名排序，跳过临时文件）。"""
    docs_dir = Path(docs_dir or USER_DOCS_DIR)
    ensure_dirs()
    if not docs_dir.exists():
        return []
    files: List[Path] = []
    for p in sorted(docs_dir.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if p.name.startswith("~$") or p.name.startswith("."):  # Office 临时文件 / 隐藏文件
            continue
        files.append(p)
    return files


def prepare_chunks(chunk_size: int = 800, chunk_overlap: int = 150,
                   docs_dir: Optional[Path] = None) -> LoadReport:
    """
    扫描并切片全部文档，返回 LoadReport。
    单个文件异常（损坏 PDF、乱码 TXT、空文件）只记录错误并跳过，不影响其它文件。
    """
    report = LoadReport()
    files = scan_user_docs(docs_dir)
    report.files_total = len(files)
    for path in files:
        try:
            chunks = load_and_split(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            report.chunks.extend(chunks)
            report.files_ok += 1
            logger.info("已解析 %s，切片 %d 条", path.name, len(chunks))
        except DocumentLoadError as exc:
            report.add_error(path.name, str(exc))
        except Exception as exc:  # 兜底：任何未预期异常也不中断
            report.add_error(path.name, f"未预期异常 {type(exc).__name__}: {exc}")
    return report


# ---------------------------------------------------------------------------
# 上传文件落盘（管理员功能 1）
# ---------------------------------------------------------------------------
def unique_target_path(docs_dir: Path, filename: str) -> Path:
    """
    生成不冲突的目标路径：重名时自动重命名为 xxx(1).后缀，绝不覆盖原文件。
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = docs_dir / f"{stem}{suffix}"
    idx = 1
    while candidate.exists():
        candidate = docs_dir / f"{stem}({idx}){suffix}"
        idx += 1
    return candidate


def _as_source_path(item: Any) -> Path:
    """
    把上传项统一解析为「源文件路径」。

    【本次修复的核心 BUG】旧实现是 `src = Path(getattr(item, "name", item))`，
    但 main.py 传入的是 pathlib.Path 对象，而 Path **自带 .name 属性**（值为文件名 basename），
    于是绝对路径被截断成相对文件名，exists() 改按服务进程的当前工作目录判断：
      - 目录下没有同名文件 -> 一律报「文件不可读」（pdf/docx/txt 全中招）；
      - 目录下恰好有同名文件 -> 会拷贝错误的对象（数据串档）。
    现在改为：路径类对象（str / os.PathLike，含 Path）原样使用；
    只有真正的「文件对象」（如 Gradio 返回的带 .name 的临时文件对象）才取 .name。
    """
    if isinstance(item, (str, os.PathLike)):
        return Path(item)
    name = getattr(item, "name", None)
    if isinstance(name, str) and name:
        return Path(name)
    raise DocumentLoadError(f"无法识别的上传项（{type(item).__name__}），已跳过")


def save_uploaded_files(file_paths: Optional[Iterable[Any]],
                        docs_dir: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """
    把上传的临时文件复制到 user_docs。
    返回 (成功保存的文件名列表, 跳过说明列表)，跳过原因按类型区分（不存在/类型不符/占用/写入失败）。

    注意：仅复制文件到 user_docs，**不会**触发向量库更新；
         需管理员手动执行「重建知识库」后新文件才会切片入库。
    """
    docs_dir = Path(docs_dir or USER_DOCS_DIR)
    ensure_dirs()
    docs_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    skipped: List[str] = []
    for item in file_paths or []:
        try:
            src = _as_source_path(item)
        except DocumentLoadError as exc:
            skipped.append(str(exc))
            continue

        name = src.name                     # 仅用于展示与落盘命名，不做路径拼接
        if not src.exists():
            logger.warning("上传源文件不存在：%s", src)
            skipped.append(f"{name}：源文件不存在或已被清理")
            continue
        if not src.is_file():
            skipped.append(f"{name}：不是普通文件")
            continue
        ext = src.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            skipped.append(f"{name}：仅允许 {'/'.join(SUPPORTED_EXTS)}")
            continue

        try:
            target = unique_target_path(docs_dir, name)
            # 二进制读写（rb/wb）：兼容中文名与含空格路径，且避免任何编码转换
            with open(src, "rb") as fsrc, open(target, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 1024 * 1024)
                fdst.flush()
                os.fsync(fdst.fileno())
            src_size, dst_size = src.stat().st_size, target.stat().st_size
            if src_size != dst_size:         # 落盘校验：防止出现 0 字节/被截断的副本
                raise OSError(f"写入后大小不一致（源 {src_size} 字节，目标 {dst_size} 字节）")
            saved.append(target.name)
            logger.info("上传文件已保存：%s（%d 字节）", target.name, dst_size)
        except PermissionError as exc:
            skipped.append(f"{name}：文件被占用或无写入权限（请关闭后重试）")
            logger.error("保存上传文件被拒绝 %s -> %s", name, exc)
        except OSError as exc:
            skipped.append(f"{name}：写入失败（{type(exc).__name__}: {exc}）")
            logger.error("保存上传文件失败 %s -> %s", name, exc)
        except Exception as exc:
            skipped.append(f"{name}：未知异常（{type(exc).__name__}: {exc}）")
            logger.exception("保存上传文件出现未预期异常：%s", name)
    return saved, skipped


# ---------------------------------------------------------------------------
# 打开原始文档目录（管理员功能 4，仅浏览）
# ---------------------------------------------------------------------------
def open_user_docs_folder(docs_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """调用系统资源管理器打开 user_docs 目录（仅浏览，不做任何文件操作）。"""
    docs_dir = Path(docs_dir or USER_DOCS_DIR)
    ensure_dirs()
    docs_dir.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(docs_dir))  # type: ignore[attr-defined]  # Windows 资源管理器
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(docs_dir)])
        else:
            subprocess.Popen(["xdg-open", str(docs_dir)])
        return True, f"已打开目录：{docs_dir}"
    except Exception as exc:
        logger.error("打开目录失败: %s", exc)
        return False, f"打开目录失败：{exc}\n目录路径：{docs_dir}"


# ---------------------------------------------------------------------------
# Agent 人设文件读写（管理员功能 6）
# ---------------------------------------------------------------------------
_GREETING_MARKERS = ("## 开场白", "开场白：", "开场白:")


def load_persona_text() -> str:
    """读取 agent_config/Person.txt 全文（文件缺失时返回空字符串）。"""
    try:
        if not PERSONA_PATH.exists():
            return ""
        return PERSONA_PATH.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error("读取人设文件失败: %s", exc)
        return ""


def save_persona_text(text: str) -> Tuple[bool, str]:
    """保存人设文件（覆盖原文件）。"""
    text = text or ""
    if len(text) > PERSONA_MAX_CHARS:
        return False, f"人设内容过长（{len(text)} 字符），请控制在 {PERSONA_MAX_CHARS} 以内"
    if not text.strip():
        return False, "人设内容不能为空"
    try:
        PERSONA_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PERSONA_PATH.with_name(f".Person.{uuid.uuid4().hex}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, PERSONA_PATH)
        logger.info("Agent 人设已更新，%d 字符", len(text))
        return True, f"人设已保存（{len(text)} 字符），下一次提问即生效"
    except Exception as exc:
        logger.error("保存人设失败: %s", exc)
        return False, f"保存失败：{exc}"


def load_persona() -> Tuple[str, str]:
    """
    解析人设文件，返回 (人设正文, 开场白)。
    开场白取 "## 开场白" 之后的文本；未配置时使用默认欢迎语。
    """
    text = load_persona_text()
    default_greeting = (
        "你好，我是本地知识库助手。请直接输入你的问题，"
        "我会检索知识库资料后作答，并在回答末尾给出参考资料置信度。"
    )
    if not text.strip():
        return "", default_greeting
    for marker in _GREETING_MARKERS:
        idx = text.find(marker)
        if idx >= 0:
            persona = text[:idx].strip()
            greeting = text[idx + len(marker):].strip()
            return persona, (greeting or default_greeting)
    return text.strip(), default_greeting


def persona_updated_at() -> str:
    """返回人设文件最后修改时间（界面展示用）。"""
    try:
        if PERSONA_PATH.exists():
            from datetime import datetime
            return datetime.fromtimestamp(PERSONA_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return "未知"
