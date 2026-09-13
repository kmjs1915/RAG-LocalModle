
**FastAPI + Chroma 本地私有化 RAG 知识库** —— 使用 BGE 系列 Embedding / Reranker 模型，自带 Web 前端，支持文档上传、知识库重建与文档问答。

- **检索链路完全本地**：文档解析、文本切片、向量化（BGE Embedding）、余弦召回（Chroma）、重排序（BGE Reranker）全部在本机完成，文档与向量数据不出本机。
- **回答生成调用大模型 API**：最终回答由大模型生成，支持 **DeepSeek** 与 **GML（智谱 GLM，glm-4-flash）** 两个 OpenAI 兼容接口，需自行配置 API Key（见「⚠️ 重要说明」）。

## ✨ 项目特性

- 🔒 **私有化部署**：文档、向量索引、模型全部本地存储，检索过程不依赖任何第三方在线服务。
- 🧠 **BGE 系列模型**：本地运行 `bge-small-zh-v1.5` Embedding 模型 + `bge-reranker-base` 重排序模型，中文检索效果好；支持 GPU（CUDA）自动加速。
- 🗂️ **Chroma 向量数据库**：轻量级本地向量库（余弦距离度量），持久化存储，无需额外部署数据库服务。
- 🌐 **自带 Web 前端**：原生 HTML/CSS/JS，由 FastAPI 直接托管，无需 Node.js 构建流程。
- 🔐 **双角色 JWT 鉴权**：管理员 / 游客两种角色，游客免密进入、仅可问答，管理功能后端强制校验。
- 📄 **文档管理闭环**：网页上传文档 → 一键重建知识库（带进度展示）→ 在线问答，全流程可视化操作。
- 📊 **置信度展示**：回答末尾自动标注「参考资料置信度」，低置信度时给出明确提示；前端可展开查看召回 Top10 与 Reranker 分数（调试信息）。
- 🚀 **一键启动**：Windows 下提供 `Start-RAG.ps1` 一键启动脚本。
- 📡 **局域网访问**：启动后控制台自动显示局域网访问地址（形如 `http://192.168.x.x:4060`），同一局域网内的手机 / 平板 / 其他电脑可直接访问，适合团队内网共享知识库。

## 🖼️ 界面预览

**登录界面**（模式选择页：管理员登录 / 普通游客免密进入）：

<img width="1917" height="951" alt="屏幕截图 2026-09-13 161750" src="https://github.com/user-attachments/assets/f6dd5ac7-6257-47d4-9767-8285df64bc98" />

**管理员界面**（左侧为管理员控制台：上传文件、访问日志、API 密钥管理、编辑人设、重建知识库等，右侧为问答区）：

<img width="1912" height="952" alt="屏幕截图 2026-09-13 161820" src="https://github.com/user-attachments/assets/0ac890c3-514c-4511-9c49-71442a978371" />

**游客界面**（仅对话功能，无管理侧边栏）：

<img width="1909" height="943" alt="屏幕截图 2026-09-13 161833" src="https://github.com/user-attachments/assets/dc55b2ca-27f3-455b-8fc6-a6ff048da30d" />


## 📁 项目目录结构

> 目录树已区分三类：**【仓库自带】**（随 GitHub 仓库提供）、**【需自行准备】**（体积大 / 含隐私，不在仓库，需使用者手动创建或下载）、**【自动生成】**（程序运行后自动创建）。

```text
RAG-LocalModle/
│
├─ 📦 仓库自带（随 GitHub 仓库提供）
│  ├─ assets/                    # 界面截图（README「界面预览」用图：login/admin/guest.png）
│  ├─ Configs/                    # 配置文件目录
│  │  ├─ config.json              # 应用配置：LLM 供应商/API、RAG 切片与检索参数、模型路径、端口
│  │  ├─ security.json            # 管理员账号与密码哈希（bcrypt）
│  │  └─ .secret.key              # API Key 加密主密钥（首次运行自动生成，不入库、不明文写配置）
│  ├─ Database/                   # 日志目录（运行时自动写入）
│  │  ├─ access_log.jsonl         # 访问日志（仅记录提问行为，不记录回答内容）
│  │  └─ app.log                  # 服务运行日志
│  ├─ agent_config/
│  │  └─ Person.txt               # Agent 人设与开场白（可在管理界面在线编辑）
│  ├─ frontend/                   # Web 前端（原生 HTML/CSS/JS，无需构建）
│  │  ├─ index.html               # 模式选择页（管理员登录 / 游客免密进入）
│  │  ├─ chat.html                # 对话主界面（侧边栏 + 聊天区）
│  │  ├─ css/main.css             # 前端样式
│  │  └─ js/                      # api.js（接口封装+JWT）、login.js、chat.js
│  ├─ lib/                        # 核心库
│  │  ├─ embedding.py             # BGE Embedding 封装（query:/passage: 前缀、L2 归一化、GPU fp16）
│  │  ├─ reranker.py              # BGE Reranker 封装（Sigmoid 0~1 打分）
│  │  ├─ chroma_kb.py             # Chroma 向量库：入库 / 余弦召回 Top10 / 重排 Top3 / 重建 / 文件锁
│  │  ├─ file_loader.py           # 文档解析（PDF/TXT/DOCX）与滑动窗口切片
│  │  ├─ llm_client.py            # DeepSeek / GML(智谱 GLM) 调用、API Key 两段式校验、RAG prompt 组装
│  │  ├─ score_eval.py            # 置信度计算、空召回/低置信兜底、问答流程编排
│  │  └─ security.py              # 配置读写、密码哈希、API Key 密文存储、访问日志
│  ├─ main.py                     # FastAPI 入口：REST 接口 + 静态页面托管（默认端口 4060）
│  ├─ Start-RAG.ps1               # Windows 一键启动脚本（自动激活 .venv 并启动服务）
│  ├─ requirements.txt            # Python 依赖清单
│  ├─ .gitignore                  # Git 忽略规则（已忽略 .venv/models/chroma_db/user_docs 等）
│  └─ LICENSE                     # MIT 开源协议
│
├─ 📥 需自行准备（不在仓库中）
│  ├─ .venv/                      # Python 虚拟环境（本地自行创建，见「部署运行步骤」）
│  └─ models/                     # BGE 模型目录（需手动下载模型放入，见「模型下载说明」）
│     ├─ bge-small-zh-v1.5/       # Embedding 向量模型（默认）
│     └─ bge-reranker-base/       # Reranker 重排序模型（默认）
│
└─ 🔁 运行后自动生成
   ├─ chroma_db/                  # Chroma 向量数据库持久化目录
   └─ user_docs/                  # 用户上传文档存放目录
```

| 目录 / 文件 | 是否在仓库中 | 来源 |
| --- | --- | --- |
| `Configs` / `Database` / `agent_config` / `frontend` / `lib`、`main.py`、`Start-RAG.ps1`、`requirements.txt`、`.gitignore`、`LICENSE` | ✅ 是 | 随仓库提供 |
| `.venv/` | ❌ 否 | 使用者本地创建虚拟环境 |
| `models/` | ❌ 否 | 使用者手动下载 BGE 模型放入 |
| `chroma_db/` | ❌ 否 | 程序运行后自动生成 |
| `user_docs/` | ❌ 否 | 程序运行后自动生成 |

## 📋 环境要求

- **操作系统**：Windows（推荐，提供 `Start-RAG.ps1` 一键启动脚本）；Linux / macOS 可直接运行 `python main.py`。
- **Python**：3.10 及以上（开发环境为 3.12）。
- **内存**：需加载 Embedding + Reranker 两个本地模型，建议 **8 GB 及以上**。
- **磁盘**：模型文件合计约 1~2 GB，另需预留文档与向量库空间。
- **GPU（可选）**：支持 CUDA 时自动使用 GPU + fp16 加速，无 GPU 自动回退 CPU。

## 🚀 部署运行步骤

### 1. 克隆仓库

```bash
git clone https://github.com/kmjs1915/RAG-LocalMode.git
cd RAG-LocalMode
```

> 说明：仓库在 GitHub 上的实际名称为 `RAG-LocalMode`（私有仓库，需拥有访问权限）。

### 2. 创建虚拟环境

Windows（PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> ⚠️ **注意**：当前仓库的 `requirements.txt` 主要包含打包 / 桌面端相关依赖（PySide6、PyInstaller 等），**并非 RAG 服务的最小运行依赖**。服务实际运行还需以下核心依赖（务必一并安装）：`fastapi`、`uvicorn`、`chromadb`、`PyJWT`、`bcrypt`、`cryptography`、`requests`、`numpy`、`pdfplumber`、`python-docx`、`torch`、`transformers`、`FlagEmbedding`。
>
> 如需 GPU 加速，请按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装与显卡 / CUDA 版本匹配的 `torch`。

### 4. 模型下载说明（⚠️ 重点）

`models/` 文件夹**不在仓库中**（模型体积大、有独立许可证），需要你手动下载并放入项目根目录的 `models/` 文件夹下。

**推荐模型**（与 `Configs/config.json` 中的默认配置一致）：

| 用途 | 推荐模型 | 说明 |
| --- | --- | --- |
| Embedding（向量化） | `BAAI/bge-small-zh-v1.5` | 默认模型，中文小模型，体积小、速度快，适合大多数场景 |
| Embedding（更高精度） | `BAAI/bge-base-zh-v1.5` / `bge-large-zh-v1.5` | 精度更高，体积与资源占用更大（可选） |
| Reranker（重排序） | `BAAI/bge-reranker-base` | 默认模型，中文重排序，显著提升检索质量 |
| Reranker（更高精度） | `BAAI/bge-reranker-large` | 精度更高，资源占用更大（可选） |

**下载方式**（任选其一）：

```bash
# 方式一：huggingface-cli
pip install -U huggingface_hub
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5
huggingface-cli download BAAI/bge-reranker-base --local-dir models/bge-reranker-base

# 方式二：ModelScope（国内下载更快）
pip install modelscope
modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir models/bge-small-zh-v1.5
modelscope download --model BAAI/bge-reranker-base --local_dir models/bge-reranker-base
```

也可以直接从 [Hugging Face - BAAI](https://huggingface.co/BAAI) 或 [ModelScope - BAAI](https://modelscope.cn/organization/BAAI) 页面手动下载模型文件。

**放置路径**（必须与配置一致）：

```text
models/
├─ bge-small-zh-v1.5/      # Embedding 模型（对应 config.json 中 embedding.model_path）
└─ bge-reranker-base/      # Reranker 模型（对应 config.json 中 reranker.model_path）
```

每个模型文件夹内需包含完整文件：`config.json`、权重文件（`model.safetensors` 或 `pytorch_model.bin`）以及分词器文件。程序启动时会自动自检模型目录并报告缺失项。

> 💡 模型路径可在 `Configs/config.json` 中修改（`embedding.model_path`、`reranker.model_path`）。**更换 Embedding 模型或修改切片参数后，请重建知识库**，不要将不同向量模型的向量混入同一集合。

### 5. 启动项目

**Windows 一键启动（推荐）**：

```powershell
.\Start-RAG.ps1
```

脚本会自动激活 `.venv` 虚拟环境并启动服务。注意：脚本内的项目路径写死为 `E:\RAG-LocalEmbedding`，若克隆到其他位置，请将脚本中的 `$ProjectRoot` 改为你的实际路径，或改用下面的手动启动方式。

**手动启动**：

```bash
python main.py
```

启动后访问（本机）：

- 模式选择页：<http://127.0.0.1:4060/>
- 对话主界面：<http://127.0.0.1:4060/chat.html>
- Swagger 接口文档：<http://127.0.0.1:4060/api/docs>

**🌐 局域网访问**：启动后，启动脚本 / `main.py` 会在控制台打印本机局域网 IP，并显示「局域网访问」地址（形如 `http://192.168.x.x:4060`）。同一局域网内的其他设备（手机、平板、其他电脑）用浏览器打开该地址即可直接使用，无需额外配置。若局域网设备无法访问，请依次检查：

1. Windows 防火墙是否放行 `4060` 端口（TCP 入站规则）；
2. 安全软件 / 杀毒软件是否拦截了 Python 的入站连接；
3. 访问设备与服务器是否处于同一局域网（注意访客网络与 AP 隔离）。

服务默认监听 `0.0.0.0:4060`；如需仅本机访问，可运行 `python main.py --host 127.0.0.1`。

**首次登录**：管理员账号 `admin`，初始密码 `abc123456`（首次运行自动创建），登录后请立即在「修改密码」中更换。

## 📝 使用说明

### 0. 配置 LLM API Key（首次问答前必做）

问答的最终回答由大模型生成，需要先配置 API Key：

1. 以管理员身份登录，进入侧边栏「更改 API-KEY」；
2. 选择供应商 **DeepSeek** 或 **GML（智谱 GLM）**，填入 API Key；
3. 点击「连通测试并保存」——系统会先对临时密钥做连通测试，通过后加密写入磁盘，并回读磁盘做二次校验，两段都通过才算保存成功；
4. 可随时切换当前启用的供应商。密钥以密文存储（`Configs/config.json`），加密主密钥 `Configs/.secret.key` 首次运行自动生成。

未配置 API Key 时，问答接口会返回友好提示「未配置 API Key，请管理员在『更改 API-KEY』中配置后再试」，但文档上传、知识库构建等其余功能不受影响。

### 1. 上传文档

1. 以管理员身份登录，进入侧边栏「上传文档」；
2. 支持 `.pdf` / `.txt` / `.docx` 格式，单文件不超过 100 MB，可多选；
3. 文件保存到 `user_docs/` 目录（重名自动命名为 `xxx(1).后缀`，不覆盖原文件）。

> 上传只保存原始文档，**不会自动更新向量库**，需执行下一步「重建知识库」后新文档才会生效。

### 2. 构建知识库

1. 在侧边栏点击「重建知识库」；
2. 系统自动完成：扫描 `user_docs/` → 解析文档 → 文本切片（默认 800 字符 / 重叠 150）→ BGE 向量化 → 写入 Chroma（`chroma_db/`），页面实时展示进度；
3. 重建完成后可在侧边栏查看知识库状态（切片数、来源文件数）。

> 手动向 `user_docs/` 增删文件后，同样需要执行「重建知识库」才会同步到向量库。

### 3. 文档问答

1. 在对话页输入问题（管理员 / 游客均可问答，游客免密进入）；
2. 系统流程：问题清洗 → BGE 向量化 → Chroma 余弦召回 Top10 → BGE Reranker 重排 → 取 Top3 → 计算置信度 → 组装「人设 + 参考资料」Prompt → 调用大模型生成回答；
3. 回答末尾标注「参考资料置信度」（基于 Top3 片段的原始余弦相似度均值），低置信度时追加提示「知识库未找到高度匹配资料，回答仅供参考」；
4. 前端可展开调试信息，查看召回 Top10 的 Reranker 分数、余弦相似度与来源文件，便于核对检索质量。

## ⚠️ 重要说明

### 为什么仓库中缺少这几个文件夹？

克隆仓库后你会发现项目里**没有** `.venv`、`models`、`chroma_db`、`user_docs` 这几个文件夹，这是正常现象：

| 目录 | 为什么不在仓库中 | 如何获得 |
| --- | --- | --- |
| `.venv/` | Python 虚拟环境与系统环境相关，体积大，不应提交到 Git | 按「部署运行步骤」第 2 步自行创建 |
| `models/` | BGE 模型文件体积大（约 1~2 GB）且有独立许可证，不适合放入代码仓库 | 按「模型下载说明」手动下载放入 |
| `chroma_db/` | 程序运行后自动生成的向量数据库持久化目录，内容随用户文档变化 | 首次运行 / 首次重建知识库时自动创建 |
| `user_docs/` | 存放用户上传的私有文档，属于个人数据，不应公开 | 首次运行时自动创建 |

这些目录均已加入 `.gitignore` 忽略。**部署本项目唯一必须手动完成的两步是：创建虚拟环境 + 下载模型放入 `models/`**，其余目录程序会自动生成。

### 关于 LLM 调用（DeepSeek / GML）

- 本项目的**检索链路（文档解析、切片、Embedding、Chroma 召回、Reranker 重排）完全本地运行**，不依赖任何在线服务；
- **回答生成**需要调用大模型 API，支持 **DeepSeek**（`deepseek-chat`）与 **GML（智谱 GLM，`glm-4-flash`）** 两个 OpenAI 兼容供应商，可在管理界面配置 API Key 并切换；
- API Key 以 Fernet 对称加密后存入 `Configs/config.json`，加密主密钥 `Configs/.secret.key` 首次运行自动生成且与机器绑定（换机器后旧密文无法解密，属预期行为），代码中不存在任何硬编码密钥；
- 未配置 API Key、密钥失效、网络超时、限流等场景均有友好提示文案，不会抛出堆栈。

### 安全与运维提示

- **默认管理员密码**为 `abc123456`，首次登录后请务必在「修改密码」中更换（新密码需至少 8 位且同时包含字母与数字）；
- 服务基于 **HTTP 明文传输 + JWT 鉴权**（HS256，默认 12 小时有效），**仅建议在可信局域网内使用**，请勿直接暴露到公网；
- 访问日志（`Database/access_log.jsonl`）仅记录访问者 IP、时间与提问内容，**不记录 LLM 回答内容**；
- 配置改动（API Key、RAG 参数、模型路径等）会立即生效，无需重启服务；其中 `Configs/config.json` 为唯一权威数据源，程序每次读写都从磁盘读取。

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。
