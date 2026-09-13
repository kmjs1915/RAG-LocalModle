/*
文件名：frontend/js/chat.js
功能：主界面逻辑 —— 身份校验、侧边栏（仅管理员渲染 DOM）、对话渲染（气泡 / 置信度 / 低置信提示 /
      检索详情折叠），以及 8 项管理员功能（6 个模态框 + 打开文档目录 + 退出登录）。游客不生成任何管理员 DOM 节点。
*/
'use strict';

(function () {
  if (!RAG.requireLogin()) return;

  /* ==========================================================================
     常量
     ========================================================================== */
  const MOBILE_BREAKPOINT = 860;          // 与 main.css 的 @media (max-width: 860px) 对应
  const STATUS_POLL_MS = 60000;           // 顶部栏状态刷新间隔
  const INPUT_MAX_HEIGHT = 180;           // 输入框自增高上限（px）
  const MIN_PASSWORD_LENGTH = 8;          // 与后端 lib.security.check_password_strength 一致
  const SCORE_DECIMALS = 4;
  const TOAST_LONG_MS = 5200;
  const PERCENT = 100;

  const SUGGESTIONS = [
    '这个知识库收录了哪些内容？',
    '帮我总结一下文档的核心流程',
    '文档里对安全要求是怎么规定的？'
  ];

  /* ==========================================================================
     状态与元素引用
     ========================================================================== */
  const state = { me: null, sending: false, kb: { chunk_count: 0, file_count: 0 } };

  const el = {
    messages: RAG.qs('#messages'),
    emptyState: RAG.qs('#emptyState'),
    emptyChips: RAG.qs('#emptyChips'),
    input: RAG.qs('#questionInput'),
    btnSend: RAG.qs('#btnSend'),
    btnClearChat: RAG.qs('#btnClearChat'),
    sidebar: RAG.qs('#sidebar'),
    sidebarNav: RAG.qs('#sidebarNav'),
    sidebarRole: RAG.qs('#sidebarRole'),
    sidebarKbInfo: RAG.qs('#sidebarKbInfo'),
    sidebarMask: RAG.qs('#sidebarMask'),
    btnSidebarToggle: RAG.qs('#btnSidebarToggle'),
    roleChip: RAG.qs('#roleChip'),
    kbBadge: RAG.qs('#kbBadge'),
    btnLogoutTop: RAG.qs('#btnLogoutTop'),
    composerStatus: RAG.qs('#composerStatus'),
    modalMask: RAG.qs('#modalMask'),
    modalTitle: RAG.qs('#modalTitle'),
    modalBody: RAG.qs('#modalBody'),
    modalFoot: RAG.qs('#modalFoot')
  };

  /* ==========================================================================
     通用小工具
     ========================================================================== */
  /** 创建带文本的元素（textContent 赋值，天然防 XSS） */
  function makeEl(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  /** 结果提示框（弹窗内的 .alert 风格） */
  function fillAlert(node, ok, html) {
    if (!node) return;
    node.hidden = false;
    node.className = 'alert ' + (ok ? 'alert-ok' : 'alert-error');
    node.innerHTML = html;
  }

  /** 进度条（fraction 为 0~1） */
  function setProgress(bar, fraction) {
    const inner = bar && bar.querySelector('i');
    if (inner) inner.style.width = Math.round(fraction * PERCENT) + '%';
  }

  /* ==========================================================================
     模态框（strict 模式：仅「取消」/ ✕ 可关闭，供修改密码弹窗使用）
     ========================================================================== */
  const Modal = {
    strict: false,
    open: function (title, bodyHtml, footHtml, opts) {
      const options = opts || {};
      el.modalTitle.textContent = title;
      el.modalBody.innerHTML = bodyHtml || '';
      el.modalFoot.innerHTML = footHtml || '';
      el.modalMask.hidden = false;
      Modal.strict = !!options.strict;
      RAG.qs('.modal', el.modalMask).classList.toggle('wide', !!options.wide);
    },
    close: function () {
      el.modalMask.hidden = true;
      el.modalBody.innerHTML = '';
      el.modalFoot.innerHTML = '';
      Modal.strict = false;
    },
    isOpen: function () { return !el.modalMask.hidden; }
  };

  // 统一关闭入口：带 data-close 的按钮（取消 / 关闭 / ✕）＋ 非 strict 弹窗的遮罩点击
  el.modalMask.addEventListener('click', function (ev) {
    if (ev.target.closest('[data-close]')) { Modal.close(); return; }
    if (!Modal.strict && ev.target === el.modalMask) Modal.close();
  });
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape' && Modal.isOpen() && !Modal.strict) Modal.close();
  });

  /* ==========================================================================
     对话渲染
     ========================================================================== */
  function hideEmptyState() {
    if (el.emptyState) { el.emptyState.remove(); el.emptyState = null; }
  }

  function scrollToBottom() {
    el.messages.scrollTop = el.messages.scrollHeight;
  }

  /** 追加一行消息，返回气泡元素（AI 气泡在左、用户气泡在右） */
  function appendRow(role) {
    hideEmptyState();
    const isUser = role === 'user';
    const row = makeEl('div', 'msg-row ' + (isUser ? 'user' : 'ai'));
    const avatar = makeEl('div', 'avatar', isUser ? '🙋' : '🤖');
    const bubble = makeEl('div', 'bubble');
    if (!isUser) row.appendChild(avatar);
    row.appendChild(bubble);
    if (isUser) row.appendChild(avatar);
    el.messages.appendChild(row);
    scrollToBottom();
    return bubble;
  }

  /** 纯文本气泡（开场白 / 用户提问） */
  function appendTextBubble(role, text) {
    const bubble = appendRow(role);
    bubble.appendChild(makeEl('div', 'bubble-text', text));
    return bubble;
  }

  /** AI 气泡：正文 + 置信度行 + 低置信提示 + 可折叠检索详情 */
  function appendAiMessage(data) {
    const bubble = appendRow('ai');
    bubble.appendChild(makeEl('div', 'bubble-text', data.answer || '（未生成回答）'));

    if (data.error) {
      const errLine = makeEl('div', 'conf-tip', '⚠️ ' + data.error);
      errLine.style.color = 'var(--warn)';
      bubble.appendChild(errLine);
    }

    const recalled = (data.debug && data.debug.recalled) || [];

    if (data.no_result) {
      bubble.appendChild(makeEl('div', 'conf-tip',
        '（知识库暂无可用资料或未检索到相关内容：请管理员上传文档后执行「重建知识库」）'));
    } else {
      const confidenceText = data.confidence_text
        || ('参考资料置信度：' + Number(data.confidence || 0).toFixed(1) + '%');
      bubble.appendChild(makeEl('div', 'conf-line' + (data.low_confidence ? ' low' : ''),
        '📊 ' + confidenceText));
      if (data.tips) {
        bubble.appendChild(makeEl('div', 'conf-tip', '⚠️ ' + data.tips));
      }
    }

    if (recalled.length) bubble.appendChild(buildDebugPanel(data.debug || {}, recalled));

    bubble.appendChild(makeEl('div', 'msg-meta', '耗时 ' + (data.elapsed_ms || 0) + ' ms'
      + (data.provider ? ' ｜ 模型 ' + data.provider : '')
      + ' ｜ 召回 ' + recalled.length + ' 条'
      + (data.rerank_ok === false ? ' ｜ Reranker 异常（已降级）' : '')));

    scrollToBottom();
    return bubble;
  }

  /** 检索详情折叠面板（召回 Top10 与 Reranker 分数 / 余弦相似度） */
  function buildDebugPanel(debug, recalled) {
    const details = makeEl('details', 'debug');
    details.appendChild(makeEl('summary', null,
      '🔍 检索详情（召回 Top' + recalled.length + ' ｜ Reranker 分数 ｜ 余弦相似度）'));

    const bodyWrap = makeEl('div', 'debug-body');
    const table = makeEl('table', 'data');
    table.innerHTML = '<thead><tr><th>重排</th><th>召回</th><th>Reranker</th><th>余弦</th>' +
      '<th>来源</th><th>片段摘要</th></tr></thead>';

    const score = function (value) { return Number(value || 0).toFixed(SCORE_DECIMALS); };
    const rank = function (value) { return value === null || value === undefined ? '-' : String(value); };

    const tbody = makeEl('tbody');
    recalled.forEach(function (c) {
      const tr = makeEl('tr');
      tr.innerHTML =
        '<td class="num">' + RAG.escapeHtml(rank(c.rerank_rank)) + '</td>' +
        '<td class="num">' + RAG.escapeHtml(rank(c.recall_rank)) + '</td>' +
        '<td class="num">' + score(c.rerank_score) + '</td>' +
        '<td class="num">' + score(c.similarity) + '</td>' +
        '<td>' + RAG.escapeHtml(c.source) + '</td>' +
        '<td class="q-cell">' + RAG.escapeHtml((c.preview || '').replace(/\n/g, ' ')) + '…</td>';
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    bodyWrap.appendChild(table);

    const avg = debug.confidence_avg === null || debug.confidence_avg === undefined
      ? '-' : Number(debug.confidence_avg).toFixed(SCORE_DECIMALS);
    const foot = makeEl('div', 'small muted',
      '置信度算法：Top3 原始余弦相似度中 ≥0.5 的算术平均（当前均值 ' + avg + '） ｜ 送入 LLM 的片段：' +
      (debug.used || []).map(function (u) { return '《' + u.source + '》#' + u.chunk_index; }).join('、'));
    foot.style.marginTop = '8px';
    bodyWrap.appendChild(foot);

    details.appendChild(bodyWrap);
    return details;
  }

  function appendTyping() {
    const bubble = appendRow('ai');
    bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span> ' +
      '<span class="muted small">正在检索知识库并生成回答…</span>';
    return bubble;
  }

  /** 移除等待气泡（用户可能在等待期间清空对话，故必须判空） */
  function removeTyping(bubble) {
    const row = bubble && bubble.closest('.msg-row');
    if (row) row.remove();
  }

  /* ==========================================================================
     发送提问
     ========================================================================== */
  function setStatus(html) {
    if (el.composerStatus) el.composerStatus.innerHTML = html;
  }

  function autoResizeInput() {
    el.input.style.height = 'auto';
    el.input.style.height = Math.min(el.input.scrollHeight, INPUT_MAX_HEIGHT) + 'px';
  }

  async function send() {
    if (state.sending) return;
    const question = (el.input.value || '').trim();
    if (!question) {
      RAG.toast('请输入问题后再发送', 'warn');
      el.input.focus();
      return;
    }
    state.sending = true;
    el.btnSend.disabled = true;
    el.input.value = '';
    autoResizeInput();
    appendTextBubble('user', question);
    const typing = appendTyping();
    setStatus('<span class="spinner"></span> 检索中…');

    try {
      const data = await RAG.chat(question);
      removeTyping(typing);
      appendAiMessage(data);
      setStatus('就绪');
      if (data.low_confidence && !data.no_result) RAG.toast('置信度较低，回答仅供参考', 'warn');
    } catch (err) {
      removeTyping(typing);
      appendAiMessage({ answer: '❌ 请求失败：' + RAG.errText(err), no_result: true });
      setStatus('请求失败');
    } finally {
      state.sending = false;
      el.btnSend.disabled = false;
      el.input.focus();
    }
  }

  el.btnSend.addEventListener('click', send);
  el.input.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter' && !ev.shiftKey) {     // Enter 发送，Shift+Enter 换行
      ev.preventDefault();
      send();
    }
  });
  el.input.addEventListener('input', autoResizeInput);

  /* ==========================================================================
     空状态与建议问题
     ========================================================================== */
  const EMPTY_STATE_HTML =
    '<div class="empty-emoji">💬</div>' +
    '<p class="empty-title">开始你的第一个提问</p>' +
    '<p class="empty-desc">我会检索本地知识库并给出带置信度的回答</p>' +
    '<div class="empty-chips" id="emptyChips"></div>';

  function renderChips() {
    if (!el.emptyChips) return;
    el.emptyChips.innerHTML = '';
    SUGGESTIONS.forEach(function (text) {
      const chip = makeEl('button', 'chip', text);
      chip.type = 'button';
      chip.addEventListener('click', function () {
        el.input.value = text;
        autoResizeInput();
        send();
      });
      el.emptyChips.appendChild(chip);
    });
  }

  el.btnClearChat.addEventListener('click', function () {
    el.messages.innerHTML = '';
    const empty = makeEl('div', 'empty-state');
    empty.id = 'emptyState';
    empty.innerHTML = EMPTY_STATE_HTML;
    el.messages.appendChild(empty);
    el.emptyState = empty;
    el.emptyChips = RAG.qs('#emptyChips');
    renderChips();
    RAG.toast('对话已清空', 'ok');
  });

  /* ==========================================================================
     侧边栏（仅管理员渲染 DOM）
     ========================================================================== */
  const ADMIN_MENU = [
    { id: 'upload', icon: '📁', label: '上传文件', handler: openUploadModal },
    { id: 'logs', icon: '📊', label: '访问日志', handler: openLogsModal },
    { id: 'apikey', icon: '🔑', label: 'API密钥管理', handler: openApiKeyModal },
    { id: 'opendocs', icon: '📂', label: '打开文档目录', handler: handleOpenDocs },
    { id: 'password', icon: '🔒', label: '修改密码', handler: openPasswordModal },
    { id: 'persona', icon: '✏️', label: '编辑人设', handler: openPersonaModal },
    { id: 'reinit', icon: '🔄', label: '重建知识库', handler: openReinitModal },
    { id: 'logout', icon: '🚪', label: '退出登录', handler: handleLogout, danger: true }
  ];

  function buildSidebar() {
    el.sidebarNav.innerHTML = '';                 // 游客不会走到这里
    ADMIN_MENU.forEach(function (item) {
      const btn = makeEl('button', 'nav-item' + (item.danger ? ' danger' : ''));
      btn.type = 'button';
      btn.dataset.id = item.id;
      btn.innerHTML = '<span class="nav-ico">' + item.icon + '</span><span>' + item.label + '</span>';
      btn.addEventListener('click', function () {
        setActive(item.id);
        item.handler();
      });
      el.sidebarNav.appendChild(btn);
    });
    el.sidebar.hidden = false;
  }

  function setActive(id) {
    RAG.qsa('.nav-item', el.sidebarNav).forEach(function (btn) {
      btn.classList.toggle('active', btn.dataset.id === id);
    });
  }

  function closeSidebarOnMobile() {
    if (window.innerWidth <= MOBILE_BREAKPOINT) {
      el.sidebar.classList.remove('open');
      el.sidebarMask.hidden = true;
    }
  }

  el.btnSidebarToggle.addEventListener('click', function () {
    el.sidebar.classList.toggle('open');
    el.sidebarMask.hidden = !el.sidebar.classList.contains('open');
  });
  el.sidebarMask.addEventListener('click', function () {
    el.sidebar.classList.remove('open');
    el.sidebarMask.hidden = true;
  });

  /* ==========================================================================
     管理员功能 1：上传文件
     ========================================================================== */
  function openUploadModal() {
    let files = [];

    Modal.open('📁 上传文件至数据库',
      '<div class="dropzone" id="dz">' +
      '  <span class="dropzone-ico">⬆️</span>' +
      '  <div>点击选择文件，或把文件拖拽到这里</div>' +
      '  <div class="dropzone-hint">仅支持 .pdf / .txt / .docx；重名文件自动重命名为 xxx(1).后缀，不覆盖原文件</div>' +
      '  <input type="file" id="fileInput" accept=".pdf,.txt,.docx" multiple hidden />' +
      '</div>' +
      '<div class="file-list" id="fileList"></div>' +
      '<div class="progress" id="upProgress" hidden><i></i></div>' +
      '<div class="progress-text" id="upProgressText" hidden></div>' +
      '<div class="alert alert-warn" id="upResult" hidden style="margin-top:14px"></div>',
      '<button class="btn btn-ghost" type="button" id="upCancel" data-close>关闭</button>' +
      '<button class="btn btn-primary" type="button" id="upStart">开始上传</button>');

    const dz = RAG.qs('#dz');
    const input = RAG.qs('#fileInput');
    const list = RAG.qs('#fileList');
    const bar = RAG.qs('#upProgress');
    const barText = RAG.qs('#upProgressText');
    const result = RAG.qs('#upResult');
    const btnStart = RAG.qs('#upStart');

    function renderFiles() {
      list.innerHTML = '';
      files.forEach(function (f, idx) {
        const row = makeEl('div', 'file-item');
        row.innerHTML = '<span>📄</span><span class="file-name">' + RAG.escapeHtml(f.name) + '</span>' +
          '<span class="file-size">' + RAG.formatBytes(f.size) + '</span>' +
          '<span class="file-del" title="移除">✕</span>';
        row.querySelector('.file-del').addEventListener('click', function () {
          files.splice(idx, 1);
          renderFiles();
        });
        list.appendChild(row);
      });
      btnStart.disabled = files.length === 0;
    }

    function addFiles(fileList) {
      Array.from(fileList).forEach(function (f) {
        const name = (f.name || '').toLowerCase();
        if (!/\.(pdf|txt|docx)$/.test(name)) {
          RAG.toast('已跳过不支持的文件：' + f.name + '（仅允许 .pdf/.txt/.docx）', 'warn');
          return;
        }
        const duplicated = files.some(function (x) { return x.name === f.name && x.size === f.size; });
        if (!duplicated) files.push(f);
      });
      renderFiles();
    }

    dz.addEventListener('click', function () { input.click(); });
    input.addEventListener('change', function () { addFiles(input.files); input.value = ''; });
    ['dragenter', 'dragover'].forEach(function (type) {
      dz.addEventListener(type, function (ev) { ev.preventDefault(); dz.classList.add('drag'); });
    });
    ['dragleave', 'drop'].forEach(function (type) {
      dz.addEventListener(type, function (ev) { ev.preventDefault(); dz.classList.remove('drag'); });
    });
    dz.addEventListener('drop', function (ev) {
      if (ev.dataTransfer && ev.dataTransfer.files) addFiles(ev.dataTransfer.files);
    });

    btnStart.addEventListener('click', async function () {
      if (!files.length) return;
      RAG.setBusy(btnStart, true, '上传中…');
      bar.hidden = false;
      barText.hidden = false;
      setProgress(bar, 0);
      barText.textContent = '正在上传…';
      try {
        const data = await RAG.upload(files, function (ratio) {
          setProgress(bar, ratio);
          barText.textContent = '上传进度 ' + Math.round(ratio * PERCENT) + '%';
        });
        const skipped = (data.skipped || []).length;
        fillAlert(result, Boolean(data.saved && data.saved.length),
          '✅ ' + RAG.escapeHtml(data.message || '上传完成') +
          (data.saved && data.saved.length ? '<br />文件：' + data.saved.map(RAG.escapeHtml).join('、') : '') +
          (skipped ? '<br />⚠️ 跳过 ' + skipped + ' 个：' + data.skipped.map(RAG.escapeHtml).join('；') : '') +
          '<br />❗' + RAG.escapeHtml(data.tip || ''));
        RAG.toast('上传完成', 'ok');
        files = [];
        renderFiles();
        loadStatus();
      } catch (err) {
        fillAlert(result, false, '❌ ' + RAG.escapeHtml(RAG.errText(err)));
      } finally {
        RAG.setBusy(btnStart, false, null, '开始上传');
        bar.hidden = true;
        barText.hidden = true;
      }
    });
  }

  /* ==========================================================================
     管理员功能 2：访问日志
     ========================================================================== */
  function openLogsModal() {
    Modal.open('📊 访问日志',
      '<div class="small muted" style="margin-bottom:10px">仅记录提问行为（访问者 IP / 最近访问时间 / ' +
      '最近一次提问 / 访问次数），不记录大模型回答。</div>' +
      '<div id="logsBox"><span class="spinner"></span> 加载中…</div>',
      '<button class="btn btn-ghost" type="button" id="logsRefresh">刷新</button>' +
      '<button class="btn btn-primary" type="button" id="logsClose" data-close>关闭</button>',
      { wide: true });

    RAG.qs('#logsRefresh').addEventListener('click', loadLogs);
    loadLogs();
  }

  async function loadLogs() {
    const box = RAG.qs('#logsBox');
    if (!box) return;
    try {
      const data = await RAG.logs();
      if (!data.rows.length) {
        box.innerHTML = '<div class="alert alert-info">暂无访问记录</div>';
        return;
      }
      box.innerHTML =
        '<div class="small muted" style="margin-bottom:8px">共 ' + data.total_users + ' 个访问者，累计 ' +
        data.total_records + ' 条提问记录</div>' +
        '<div class="table-wrap"><table class="data"><thead><tr>' +
        '<th>访问者 IP</th><th>最近访问时间</th><th>最近一次提问内容</th><th>访问次数</th>' +
        '</tr></thead><tbody>' +
        data.rows.map(function (r) {
          return '<tr><td class="mono">' + RAG.escapeHtml(r.ip) + '</td>' +
            '<td class="mono">' + RAG.escapeHtml(r.time) + '</td>' +
            '<td class="q-cell">' + RAG.escapeHtml(r.question) + '</td>' +
            '<td class="num">' + RAG.escapeHtml(r.count) + '</td></tr>';
        }).join('') + '</tbody></table></div>';
    } catch (err) {
      box.innerHTML = '<div class="alert alert-error">加载失败：' + RAG.escapeHtml(RAG.errText(err)) + '</div>';
    }
  }

  /* ==========================================================================
     管理员功能 3：API 密钥管理（仅展示，校验逻辑全在后端）
     ========================================================================== */
  let apiKeyRenderId = 0;                 // 防止过期请求把结果写进已重建的 DOM

  function openApiKeyModal() {
    Modal.open('🔑 API 密钥管理',
      '<div id="apiKeyBox"><span class="spinner"></span> 读取磁盘配置中…</div>' +
      '<div class="alert alert-info" style="margin-top:12px">' +
      '保存流程：先用输入框中的密钥做连通测试 → 通过后加密写入 Configs/config.json → ' +
      '后端立即回读磁盘做二次校验 → 全部通过才算保存成功（失败自动回滚）。密钥仅显示掩码，不显示明文。' +
      '</div>' +
      '<div id="apiKeyMsg"></div>',
      '<button class="btn btn-ghost" type="button" id="akRefresh">刷新状态（实测连通）</button>' +
      '<button class="btn btn-primary" type="button" id="akClose" data-close>关闭</button>',
      { wide: true });

    RAG.qs('#akRefresh').addEventListener('click', function () { loadApiKeys(true); });
    loadApiKeys(false);
  }

  /** 单个供应商卡片（密钥输入框 + 连通测试 + 保存） */
  function providerCardHtml(provider, activeProvider) {
    const isActive = activeProvider === provider.provider;
    return '' +
      '<div class="kv-row" style="display:block">' +
      '  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px">' +
      '    <strong>' + RAG.escapeHtml(provider.label) + '</strong>' +
      '    <span class="status-dot ' + (provider.configured ? 'on' : 'off') + '">' +
      (provider.configured ? '已配置' : '未配置') + '</span>' +
      (isActive ? '<span class="role-chip" style="font-size:11px">当前启用</span>' : '') +
      '    <span class="muted small">模型 ' + RAG.escapeHtml(provider.model || '-') +
      ' ｜ 密钥 <span class="mono">' + RAG.escapeHtml(provider.mask) + '</span>' +
      ' ｜ 更新于 ' + RAG.escapeHtml(provider.updated_at || '-') + '</span>' +
      '  </div>' +
      '  <div class="form-row">' +
      '    <input class="input" type="password" id="key_' + provider.provider + '" placeholder="粘贴新的 ' +
      RAG.escapeHtml(provider.label) + '（留空可只做连通测试）" style="flex:1 1 auto" />' +
      '    <button class="btn btn-ghost" type="button" data-test="' + provider.provider + '">连通测试</button>' +
      '    <button class="btn btn-primary" type="button" data-save="' + provider.provider + '">保存</button>' +
      '  </div>' +
      '  <div class="small" data-result="' + provider.provider + '" style="margin-top:8px"></div>' +
      '</div>';
  }

  async function loadApiKeys(liveCheck) {
    const box = RAG.qs('#apiKeyBox');
    if (!box) return;
    const renderId = ++apiKeyRenderId;
    try {
      const data = await RAG.apikeyStatus();
      if (renderId !== apiKeyRenderId) return;          // 已有更新的渲染，丢弃本次结果

      box.innerHTML = data.providers.map(function (p) {
        return providerCardHtml(p, data.active_provider);
      }).join('') + '<div class="small muted">当前启用的大模型：<span class="mono">' +
        RAG.escapeHtml(data.active_provider) + '</span></div>';

      RAG.qsa('[data-test]', box).forEach(function (btn) {
        btn.addEventListener('click', function () {
          const key = (RAG.qs('#key_' + btn.dataset.test).value || '').trim();
          runApiKeyTest(btn.dataset.test, key, btn, renderId);
        });
      });

      RAG.qsa('[data-save]', box).forEach(function (btn) {
        btn.addEventListener('click', function () { runApiKeySave(btn.dataset.save, btn, renderId); });
      });

      if (liveCheck) {
        // 「刷新状态（实测连通）」：对已配置的供应商用磁盘密钥实测一次
        for (const provider of data.providers) {
          if (renderId !== apiKeyRenderId) return;
          if (!provider.configured) continue;
          await runApiKeyTest(provider.provider, '', null, renderId);
        }
      }
    } catch (err) {
      if (renderId === apiKeyRenderId) {
        box.innerHTML = '<div class="alert alert-error">读取失败：' + RAG.escapeHtml(RAG.errText(err)) + '</div>';
      }
    }
  }

  /** 连通测试（传 key 则测临时密钥，留空则测磁盘里的密钥） */
  async function runApiKeyTest(provider, key, btn, renderId) {
    setProviderResult(provider, '<span class="spinner"></span> ' +
      (key ? '正在连通测试…' : '读取磁盘密钥实测中…'), '');
    if (btn) btn.disabled = true;
    try {
      const res = await RAG.apikeyTest(provider, key);
      if (renderId === apiKeyRenderId) {
        setProviderResult(provider, (res.ok ? '✅ ' : '❌ ') + RAG.escapeHtml(res.message),
          res.ok ? 'on' : 'off');
      }
    } catch (err) {
      if (renderId === apiKeyRenderId) {
        setProviderResult(provider, '❌ ' + RAG.escapeHtml(RAG.errText(err)), 'off');
      }
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  /** 保存密钥（后端两段式校验，失败自动回滚） */
  async function runApiKeySave(provider, btn, renderId) {
    const input = RAG.qs('#key_' + provider);
    const key = (input.value || '').trim();
    if (!key) {
      RAG.toast('请先粘贴要保存的 API Key', 'warn');
      input.focus();
      return;
    }
    btn.disabled = true;
    setProviderResult(provider, '<span class="spinner"></span> 连通测试 → 写入磁盘 → 回读二次校验…', '');

    let line = '';
    let dot = '';
    try {
      const res = await RAG.apikeySave(provider, key, true);
      input.value = '';
      line = '✅ ' + RAG.escapeHtml(res.message || '保存成功');
      dot = 'on';
      RAG.toast('API Key 保存成功并通过二次校验', 'ok');
      loadStatus();
    } catch (err) {
      line = '❌ ' + RAG.escapeHtml(RAG.errText(err));
      dot = 'off';
      RAG.toast('保存失败，配置未变更', 'err');
    } finally {
      btn.disabled = false;
      await loadApiKeys(false);        // 重新读磁盘状态（会重建 DOM，故之后再写入结果）
      setProviderResult(provider, line, dot);
    }
  }

  /** 更新某个供应商的结果文案与连通状态点 */
  function setProviderResult(provider, html, dotClass) {
    const node = RAG.qs('[data-result="' + provider + '"]');
    if (node) node.innerHTML = html;
    if (!dotClass || !node) return;
    const row = node.closest('.kv-row');
    const dot = row && row.querySelector('.status-dot');
    if (!dot) return;
    dot.classList.remove('on', 'off');
    dot.classList.add(dotClass);
    dot.textContent = dotClass === 'on' ? '已连接' : '未连接';
  }

  /* ==========================================================================
     管理员功能 4：打开原始文档目录
     ========================================================================== */
  async function handleOpenDocs() {
    try {
      const res = await RAG.openDocs();
      RAG.toast((res.ok ? '✅ ' : '❌ ') + (res.message || ''), res.ok ? 'ok' : 'err');
      if (res.tip) RAG.toast(res.tip, 'warn', TOAST_LONG_MS);
    } catch (err) {
      RAG.toast('❌ ' + RAG.errText(err), 'err');
    } finally {
      closeSidebarOnMobile();
    }
  }

  /* ==========================================================================
     管理员功能 5：修改密码（strict 弹窗：仅「取消」/ ✕ 可关闭，报错不关闭）
     ========================================================================== */
  const PASSWORD_FIELDS = ['#oldPwd', '#newPwd', '#confirmPwd'];

  function openPasswordModal() {
    Modal.open('🔒 修改密码',
      '<div class="field"><label class="field-label" for="oldPwd">旧密码</label>' +
      '<input class="input" type="password" id="oldPwd" autocomplete="current-password" /></div>' +
      '<div class="field"><label class="field-label" for="newPwd">新密码（至少 8 位，需含字母与数字）</label>' +
      '<input class="input" type="password" id="newPwd" autocomplete="new-password" /></div>' +
      '<div class="field"><label class="field-label" for="confirmPwd">确认新密码</label>' +
      '<input class="input" type="password" id="confirmPwd" autocomplete="new-password" /></div>' +
      '<div class="inline-error" id="pwdError" hidden></div>',
      '<button class="btn btn-ghost" type="button" id="pwdCancel" data-close>取消</button>' +
      '<button class="btn btn-primary" type="button" id="pwdSubmit">提交修改</button>',
      { strict: true });                                          // ← 仅「取消」/「✕」可关闭

    const errBox = RAG.qs('#pwdError');
    const showFormMessage = function (ok, message) {
      errBox.className = 'inline-error' + (ok ? ' ok' : '');
      errBox.textContent = (ok ? '✅ ' : '❌ ') + message;
      errBox.hidden = false;
    };
    const clearFormMessage = function () { errBox.hidden = true; errBox.textContent = ''; };

    // 输入时清除上一次提示
    PASSWORD_FIELDS.forEach(function (sel) {
      const node = RAG.qs(sel);
      if (node) node.addEventListener('input', clearFormMessage);
    });

    RAG.qs('#pwdSubmit').addEventListener('click', async function () {
      const oldPwd = RAG.qs('#oldPwd').value;
      const newPwd = RAG.qs('#newPwd').value;
      const confirmPwd = RAG.qs('#confirmPwd').value;
      clearFormMessage();
      // ---- 前端预校验：不通过时只在弹窗内提示，弹窗保持打开 ----
      if (!oldPwd || !newPwd || !confirmPwd) { showFormMessage(false, '请填写完整的三项密码'); return; }
      if (newPwd !== confirmPwd) { showFormMessage(false, '两次输入的新密码不一致'); return; }
      if (newPwd.length < MIN_PASSWORD_LENGTH) {
        showFormMessage(false, '新密码长度至少 ' + MIN_PASSWORD_LENGTH + ' 位');
        return;
      }
      if (!/[A-Za-z]/.test(newPwd) || !/[0-9]/.test(newPwd)) {
        showFormMessage(false, '新密码需同时包含字母和数字');
        return;
      }
      if (newPwd === oldPwd) { showFormMessage(false, '新密码不能与旧密码相同'); return; }

      const btn = RAG.qs('#pwdSubmit');
      RAG.setBusy(btn, true, '提交中…');
      try {
        const res = await RAG.changePassword(oldPwd, newPwd, confirmPwd);
        // 成功：清空输入并提示；弹窗保持打开（由用户点「取消」/「✕」关闭）
        showFormMessage(true, res.message || '密码修改成功，请牢记新密码');
        PASSWORD_FIELDS.forEach(function (sel) { RAG.qs(sel).value = ''; });
        RAG.toast('密码修改成功，请牢记新密码', 'ok', 5000);
      } catch (err) {
        // 失败：仅内联展示红色错误，绝不关闭弹窗
        showFormMessage(false, RAG.errText(err));
      } finally {
        RAG.setBusy(btn, false, null, '提交修改');
      }
    });
  }

  /* ==========================================================================
     管理员功能 6：编辑 Agent 人设
     ========================================================================== */
  function openPersonaModal() {
    Modal.open('✏️ 编辑 Agent 人设',
      '<div class="small muted" style="margin-bottom:8px" id="personaMeta">加载中…</div>' +
      '<textarea class="textarea" id="personaText" placeholder="正在读取 agent_config/Person.txt …"></textarea>' +
      '<div id="personaMsg"></div>',
      '<button class="btn btn-ghost" type="button" id="personaReload">重新读取</button>' +
      '<button class="btn btn-ghost" type="button" id="personaClose" data-close>关闭</button>' +
      '<button class="btn btn-primary" type="button" id="personaSave">保存人设</button>',
      { wide: true });

    RAG.qs('#personaReload').addEventListener('click', loadPersona);
    RAG.qs('#personaSave').addEventListener('click', async function () {
      const btn = RAG.qs('#personaSave');
      RAG.setBusy(btn, true, '保存中…');
      try {
        const res = await RAG.personaSave(RAG.qs('#personaText').value);
        RAG.qs('#personaMsg').innerHTML = '<div class="alert alert-ok" style="margin-top:12px">✅ ' +
          RAG.escapeHtml(res.message) + '</div>';
        RAG.toast('人设已保存，下次提问即生效', 'ok');
        loadPersona();
      } catch (err) {
        RAG.qs('#personaMsg').innerHTML = '<div class="alert alert-error" style="margin-top:12px">❌ ' +
          RAG.escapeHtml(RAG.errText(err)) + '</div>';
      } finally {
        RAG.setBusy(btn, false, null, '保存人设');
      }
    });
    loadPersona();
  }

  async function loadPersona() {
    const meta = RAG.qs('#personaMeta');
    const box = RAG.qs('#personaText');
    if (!meta || !box) return;
    try {
      const data = await RAG.personaGet();
      box.value = data.content || '';
      meta.textContent = '文件：agent_config/Person.txt ｜ ' + data.length + ' 字符 ｜ 最后修改：' +
        (data.updated_at || '-');
    } catch (err) {
      meta.textContent = '读取失败：' + RAG.errText(err);
    }
  }

  /* ==========================================================================
     管理员功能 7：重建知识库（含进度轮询）
     ========================================================================== */
  function openReinitModal() {
    Modal.open('🔄 重建知识库',
      '<div class="alert alert-info">将清空 Chroma 集合 <span class="mono">personal_kb</span>，并重新扫描 ' +
      '<span class="mono">user_docs</span> 全量切片入库。写操作已加文件锁，可防止并发损坏向量库。' +
      '文档较多时可能需要数分钟。</div>' +
      '<div class="kv-row" style="margin-top:14px">' +
      '  <span class="kv-key">当前切片数：</span><span class="kv-val" id="kbChunks">' + state.kb.chunk_count + '</span>' +
      '  <span class="kv-key">来源文件：</span><span class="kv-val" id="kbFiles">' + state.kb.file_count + '</span>' +
      '  <span class="kv-key">待入库文档：</span><span class="kv-val" id="kbPending">-</span>' +
      '</div>' +
      '<div class="progress" id="kbBar" hidden><i></i></div>' +
      '<div class="progress-text" id="kbBarText" hidden></div>' +
      '<div id="kbMsg"></div>',
      '<button class="btn btn-ghost" type="button" id="kbCancel" data-close>取消</button>' +
      '<button class="btn btn-primary" type="button" id="kbGo">确认重建</button>');

    refreshKbPending();

    RAG.qs('#kbGo').addEventListener('click', async function () {
      const btn = RAG.qs('#kbGo');
      const bar = RAG.qs('#kbBar');
      const barText = RAG.qs('#kbBarText');
      const msg = RAG.qs('#kbMsg');
      RAG.setBusy(btn, true, '执行中…');
      bar.hidden = false;
      barText.hidden = false;
      msg.innerHTML = '';

      const onProgress = function (p) {
        const percent = Math.round((p.ratio || 0) * PERCENT);
        setProgress(bar, Math.max(0.03, percent / PERCENT));
        barText.textContent = (p.message || '处理中…') + '（' + percent + '%，已用时 ' + (p.elapsed || 0) + 's）';
      };

      try {
        const res = await RAG.kbReinit(onProgress);
        msg.innerHTML = '<div class="alert alert-ok" style="margin-top:12px">✅ ' + RAG.escapeHtml(res.message) +
          (res.tip ? '<br />' + RAG.escapeHtml(res.tip) : '') + '</div>';
        RAG.toast('知识库重建完成', 'ok', 4500);
        await loadStatus();
      } catch (err) {
        msg.innerHTML = '<div class="alert alert-error" style="margin-top:12px">❌ ' +
          RAG.escapeHtml(RAG.errText(err)) + '</div>';
      } finally {
        RAG.setBusy(btn, false, null, '确认重建');
        bar.hidden = true;
        barText.hidden = true;
      }
    });
  }

  function updateKbRow() {
    const chunks = RAG.qs('#kbChunks');
    const files = RAG.qs('#kbFiles');
    if (chunks) chunks.textContent = state.kb.chunk_count;
    if (files) files.textContent = state.kb.file_count;
  }

  async function refreshKbPending() {
    try {
      const status = await RAG.status();
      const pending = RAG.qs('#kbPending');
      if (pending) {
        pending.textContent = status.docs_pending > 0
          ? status.docs_pending + ' 个（需重建后生效）'
          : '无（已全部入库）';
      }
    } catch (e) { /* 状态获取失败不影响弹窗 */ }
  }

  /* ==========================================================================
     管理员功能 9：退出登录
     ========================================================================== */
  async function handleLogout() {
    try { await RAG.logout(); } catch (e) { /* 忽略网络错误，本地照样清除 */ }
    RAG.token.clear();
    location.replace('/index.html');
  }

  el.btnLogoutTop.addEventListener('click', function () {
    if (window.confirm('确认退出登录并返回登录页？')) handleLogout();
  });

  /* ==========================================================================
     启动与状态同步
     ========================================================================== */
  function renderSidebarKbInfo(status) {
    if (el.sidebarKbInfo) {
      el.sidebarKbInfo.textContent = '知识库：' + state.kb.chunk_count + ' 切片 / '
        + state.kb.file_count + ' 文件 ｜ 模型 ' + (status.active_provider || '-');
    }
  }

  async function loadStatus() {
    try {
      const status = await RAG.status();
      state.kb = status.kb || { chunk_count: 0, file_count: 0 };
      if (el.kbBadge) {
        el.kbBadge.textContent = '切片 ' + state.kb.chunk_count + ' · 文件 ' + state.kb.file_count
          + (status.docs_pending > 0 ? ' · 待入库 ' + status.docs_pending : '');
      }
      renderSidebarKbInfo(status);
      updateKbRow();
      return status;
    } catch (err) {
      if (el.kbBadge) el.kbBadge.textContent = '状态获取失败';
      return null;
    }
  }

  /** 管理员视图：渲染侧边栏并显示移动端菜单按钮 */
  function setupAdminView() {
    el.sidebarRole.textContent = '管理员';
    el.sidebarRole.style.color = '#c7d2fe';
    buildSidebar();                                  // 仅管理员创建侧边栏 DOM
    if (window.innerWidth <= MOBILE_BREAKPOINT) el.btnSidebarToggle.hidden = false;
  }

  /** 游客视图：移除侧边栏节点，只在顶栏展示角色 */
  function setupGuestView() {
    if (el.sidebar && el.sidebar.parentNode) el.sidebar.parentNode.removeChild(el.sidebar);
    el.roleChip.textContent = '当前身份：普通游客（仅对话权限）';
  }

  function appendGreeting(status) {
    const greeting = status && status.welcome;
    if (greeting) appendTextBubble('ai', greeting);
  }

  function onWindowResize() {
    if (!state.me || !state.me.is_admin) return;
    el.btnSidebarToggle.hidden = window.innerWidth > MOBILE_BREAKPOINT;
    if (window.innerWidth > MOBILE_BREAKPOINT) {
      el.sidebar.classList.remove('open');
      el.sidebarMask.hidden = true;
    }
  }

  async function boot() {
    let me;
    try {
      me = await RAG.me();
    } catch (err) {
      // 401 已由 api.js 统一跳转；其它错误给出提示
      RAG.toast('身份校验失败：' + RAG.errText(err), 'err');
      return;
    }
    state.me = me;

    el.roleChip.textContent = '当前身份：' + me.role_label + (me.is_admin ? '（' + me.username + '）' : '');
    el.roleChip.classList.toggle('guest', !me.is_admin);
    if (me.is_admin) setupAdminView(); else setupGuestView();

    renderChips();
    appendGreeting(await loadStatus());

    el.input.focus();
    window.addEventListener('resize', onWindowResize);
    setInterval(loadStatus, STATUS_POLL_MS);         // 定时同步切片数 / 待入库文档
  }

  boot();
})();
