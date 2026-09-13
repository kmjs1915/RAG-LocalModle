/*
文件名：frontend/js/api.js
功能：统一封装后端 fetch 请求 + JWT token 管理 + 轻提示 + 通用 UI 工具。
      纯原生 ES6，无第三方依赖；login.js / chat.js 只通过 window.RAG 调用。
接口（与 main.py 一一对应）：
      POST /api/login  /api/guest/login  /api/logout   GET /api/me  /api/status
      POST /api/rag/chat
      POST /api/admin/upload  /api/admin/open-docs  /api/admin/password
      GET  /api/admin/logs  /api/admin/apikey  /api/admin/persona  /api/admin/kb/progress
      POST /api/admin/apikey  /api/admin/apikey/test  /api/admin/persona
      POST /api/admin/kb/reinit
*/
'use strict';

const RAG = (function () {
  /* ==========================================================================
     常量
     ========================================================================== */
  /**
   * token 存储策略（改这一行即可切换）：
   *   'session' → sessionStorage：同标签页刷新保留登录，关闭标签页即丢失（默认）
   *   'local'   → localStorage  ：刷新与关闭标签页都保留
   *   'memory'  → 仅内存 + URL hash 传递：刷新页面即丢失登录
   */
  const TOKEN_MODE = 'session';
  const TOKEN_KEY = 'rag_token';

  const LOGIN_PAGE = '/index.html';
  const CHAT_PAGE = '/chat.html';

  const API = {
    login: '/api/login',
    guestLogin: '/api/guest/login',
    logout: '/api/logout',
    me: '/api/me',
    status: '/api/status',
    chat: '/api/rag/chat',
    upload: '/api/admin/upload',
    logs: '/api/admin/logs',
    apikey: '/api/admin/apikey',
    apikeyTest: '/api/admin/apikey/test',
    openDocs: '/api/admin/open-docs',
    password: '/api/admin/password',
    persona: '/api/admin/persona',
    kbReinit: '/api/admin/kb/reinit',
    kbProgress: '/api/admin/kb/progress'
  };

  const AUTH_FAIL_STATUS = [401, 403];          // 需要跳回登录页的状态码
  const BEARER = 'Bearer ';
  const TOAST_MS = { normal: 3200, error: 6000, fade: 260 };
  const REDIRECT_MS = { unauthorized: 700, forbidden: 1500 };
  const KB_POLL_MS = 800;                       // 知识库长任务进度轮询间隔
  const BYTE_KB = 1024, BYTE_MB = 1024 * 1024;

  /* ==========================================================================
     通用工具
     ========================================================================== */
  function qs(selector, root) { return (root || document).querySelector(selector); }
  function qsa(selector, root) { return Array.from((root || document).querySelectorAll(selector)); }

  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n < BYTE_KB) return n + ' B';
    if (n < BYTE_MB) return (n / BYTE_KB).toFixed(1) + ' KB';
    return (n / BYTE_MB).toFixed(2) + ' MB';
  }

  /** 统一的错误文案提取，避免各处重复写 err.detail || err.message */
  function errText(err) {
    if (!err) return '未知错误';
    return err.detail || err.message || '未知错误';
  }

  function isAuthFail(status) { return AUTH_FAIL_STATUS.indexOf(status) !== -1; }

  /** 按钮忙碌态：busy=true 显示 spinner，恢复时还原原始文案（只需首次传入 idleLabel） */
  function setBusy(btn, busy, busyLabel, idleLabel) {
    if (!btn) return;
    if (busy) {
      if (!btn.dataset.idleLabel) btn.dataset.idleLabel = idleLabel || btn.textContent;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> ' + (busyLabel || '处理中…');
      return;
    }
    btn.disabled = false;
    btn.textContent = idleLabel || btn.dataset.idleLabel || btn.textContent;
  }

  /** 轻提示（自动消失） */
  function toast(message, type, timeout) {
    const wrap = qs('#toastWrap');
    if (!wrap) { console.log('[toast]', type || 'info', message); return; }
    const box = document.createElement('div');
    box.className = 'toast ' + (type || '');
    box.textContent = message;
    wrap.appendChild(box);
    const ttl = timeout || (type === 'err' ? TOAST_MS.error : TOAST_MS.normal);
    setTimeout(function () {
      box.classList.add('hide');
      setTimeout(function () { box.remove(); }, TOAST_MS.fade);
    }, ttl);
  }

  /* ==========================================================================
     token 管理
     ========================================================================== */
  let memoryToken = null;                       // memory 模式下的内存 token

  function storage() {
    if (TOKEN_MODE === 'memory') return null;
    try {
      return TOKEN_MODE === 'local' ? localStorage : sessionStorage;
    } catch (e) {
      return null;                              // 隐私模式下存储不可用，退化为内存态
    }
  }

  const TokenStore = {
    get: function () {
      if (TOKEN_MODE === 'memory') {
        // memory 模式：仅内存保存；跨页面跳转时由 URL hash 传入一次（不落任何存储）
        if (!memoryToken) {
          const fromHash = readTokenFromHash();
          if (fromHash) { memoryToken = fromHash; clearHash(); }
        }
        return memoryToken;
      }
      const store = storage();
      if (!store) return memoryToken;
      try {
        memoryToken = store.getItem(TOKEN_KEY);
        return memoryToken;
      } catch (e) {
        return memoryToken;
      }
    },
    set: function (token) {
      memoryToken = token;
      const store = storage();
      if (!store) return;
      try { store.setItem(TOKEN_KEY, token); } catch (e) { /* 退化为内存态 */ }
    },
    clear: function () {
      memoryToken = null;
      try {
        sessionStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(TOKEN_KEY);
      } catch (e) { /* ignore */ }
    }
  };

  function readTokenFromHash() {
    const hash = (location.hash || '').slice(1);
    if (!hash) return '';
    const matched = /(?:^|&)t=([^&]+)/.exec(hash);
    return matched ? decodeURIComponent(matched[1]) : '';
  }

  function clearHash() {
    if (location.hash) history.replaceState(null, '', location.pathname + location.search);
  }

  /* ==========================================================================
     请求封装
     ========================================================================== */
  function ApiError(status, detail) {
    const err = new Error(detail || ('请求失败（HTTP ' + status + '）'));
    err.name = 'ApiError';
    err.status = status;
    err.detail = detail || '';
    return err;
  }

  /** 401/403 统一处理：清除 token 并跳回登录页（silentAuth 调用点见下方请求实现） */
  function handleAuthFailure(status, detail) {
    TokenStore.clear();
    if (status === 401) {
      toast('登录已失效，请重新登录', 'warn', 1500);
    } else {
      toast(detail || '权限不足，仅管理员可执行该操作', 'err', 1800);
    }
    setTimeout(function () { location.replace(LOGIN_PAGE); },
               status === 401 ? REDIRECT_MS.unauthorized : REDIRECT_MS.forbidden);
  }

  /** 解析响应文本为 JSON（失败时退回可读文本） */
  function parseBody(text) {
    if (!text) return null;
    try { return JSON.parse(text); } catch (e) { return { detail: String(text).slice(0, 300) }; }
  }

  async function request(path, options) {
    const opt = options || {};
    const headers = Object.assign({}, opt.headers || {});
    const token = TokenStore.get();
    if (token) headers.Authorization = BEARER + token;

    let body = opt.body;
    if (body !== undefined && body !== null && !(body instanceof FormData)) {
      headers['Content-Type'] = 'application/json; charset=utf-8';
      body = JSON.stringify(body);
    }

    let resp;
    try {
      resp = await fetch(path, { method: opt.method || 'GET', headers: headers, body: body, cache: 'no-store' });
    } catch (netErr) {
      throw ApiError(0, '网络请求失败，请确认后端服务是否已启动（python main.py）');
    }

    if (resp.status === 204) return null;
    const data = parseBody(await resp.text());

    if (!resp.ok) {
      const detail = (data && (data.detail || data.message)) || ('HTTP ' + resp.status);
      // silentAuth：跳过全局跳登录页，交由调用方（如修改密码弹窗）内联提示
      if (isAuthFail(resp.status) && !opt.silentAuth) handleAuthFailure(resp.status, detail);
      throw ApiError(resp.status, typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return data;
  }

  const get = function (path, opts) {
    return request(path, Object.assign({ method: 'GET' }, opts || {}));
  };
  const post = function (path, body, opts) {
    return request(path, Object.assign({ method: 'POST', body: body }, opts || {}));
  };

  /** 触发知识库长任务，并在执行期间轮询进度接口上报进度 */
  async function kbTask(path, onProgress) {
    let timer = null;
    const poll = async function (report) {
      try {
        const progress = await get(API.kbProgress);
        if (report) onProgress(progress);
      } catch (e) { /* 轮询失败不影响主请求 */ }
    };
    if (onProgress) {
      poll(true);
      timer = setInterval(function () { poll(true); }, KB_POLL_MS);
    }
    try {
      return await post(path, {});
    } finally {
      if (timer) clearInterval(timer);
      if (onProgress) await poll(true);
    }
  }

  /** 上传（用 XHR 以获得上传进度） */
  function upload(files, onProgress) {
    return new Promise(function (resolve, reject) {
      const list = Array.from(files || []);
      if (!list.length) { reject(ApiError(400, '请选择要上传的文件')); return; }

      const form = new FormData();
      list.forEach(function (f) { form.append('files', f, f.name); });

      const xhr = new XMLHttpRequest();
      xhr.open('POST', API.upload, true);
      const token = TokenStore.get();
      if (token) xhr.setRequestHeader('Authorization', BEARER + token);
      xhr.upload.onprogress = function (ev) {
        if (onProgress && ev.lengthComputable) onProgress(ev.loaded / ev.total);
      };
      xhr.onerror = function () { reject(ApiError(0, '上传失败：网络错误')); };
      xhr.onload = function () {
        const data = parseBody(xhr.responseText) || {};
        if (xhr.status >= 200 && xhr.status < 300) { resolve(data); return; }
        const detail = data.detail || ('上传失败（HTTP ' + xhr.status + '）');
        if (isAuthFail(xhr.status)) handleAuthFailure(xhr.status, detail);
        reject(ApiError(xhr.status, detail));
      };
      xhr.send(form);
    });
  }

  /* ==========================================================================
     页面跳转
     ========================================================================== */
  function requireLogin() {
    if (TokenStore.get()) return true;
    location.replace(LOGIN_PAGE);
    return false;
  }

  /** 跳转主界面：memory 模式用 hash 传递 token（不落盘） */
  function goToChat() {
    if (TOKEN_MODE === 'memory') {
      location.replace(CHAT_PAGE + '#t=' + encodeURIComponent(TokenStore.get() || ''));
      return;
    }
    location.replace(CHAT_PAGE);
  }

  /** 登录成功后保存 token 并返回响应 */
  function saveSession(data) {
    TokenStore.set(data.token);
    return data;
  }

  /* ==========================================================================
     业务接口
     ========================================================================== */
  return {
    /* ---- 鉴权 ---- */
    login: function (username, password) {
      // silentAuth：登录失败返回 401 时不触发全局跳转/重载，
      // 由登录页在表单内展示红色错误提示（密码错误属于业务校验，不是 JWT 失效）
      return post(API.login, { username: username, password: password }, { silentAuth: true })
        .then(saveSession);
    },
    guestLogin: function () {
      return post(API.guestLogin, {}).then(saveSession);
    },
    logout: function () {
      return post(API.logout, {}).catch(function () { return { ok: true }; })
        .then(function (res) { TokenStore.clear(); return res; });
    },
    me: function () { return get(API.me); },
    status: function () { return get(API.status); },

    /* ---- 对话 ---- */
    chat: function (question) { return post(API.chat, { question: question }); },

    /* ---- 管理员 ---- */
    upload: upload,
    logs: function () { return get(API.logs); },
    apikeyStatus: function () { return get(API.apikey); },
    apikeyTest: function (provider, apiKey) {
      return post(API.apikeyTest, { provider: provider, api_key: apiKey || '' });
    },
    apikeySave: function (provider, apiKey, setActive) {
      return post(API.apikey, { provider: provider, api_key: apiKey, set_active: !!setActive });
    },
    openDocs: function () { return post(API.openDocs, {}); },
    // silentAuth：修改密码失败（含 401/403）时不跳登录页，由弹窗内联展示错误
    changePassword: function (oldPwd, newPwd, confirmPwd) {
      return post(API.password, {
        old_password: oldPwd, new_password: newPwd, confirm_password: confirmPwd
      }, { silentAuth: true });
    },
    personaGet: function () { return get(API.persona); },
    personaSave: function (content) { return post(API.persona, { content: content }); },
    kbReinit: function (onProgress) { return kbTask(API.kbReinit, onProgress); },

    /* ---- 通用工具 ---- */
    toast: toast,
    setBusy: setBusy,
    errText: errText,
    escapeHtml: escapeHtml,
    formatBytes: formatBytes,
    qs: qs,
    qsa: qsa,
    token: TokenStore,
    requireLogin: requireLogin,
    goToChat: goToChat
  };
})();
