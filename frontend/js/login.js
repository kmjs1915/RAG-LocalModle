/*
文件名：frontend/js/login.js
功能：模式选择页逻辑 —— 管理员登录表单展开/收起、游客一键登录、登录后跳转 chat.html。
      请求与通用工具统一走 api.js（RAG.*），不抛未捕获异常。
*/
'use strict';

(function () {
  const els = {
    modeSelect: RAG.qs('#modeSelect'),
    btnShowAdmin: RAG.qs('#btnShowAdmin'),
    btnGuest: RAG.qs('#btnGuest'),
    adminForm: RAG.qs('#adminForm'),
    btnBackMode: RAG.qs('#btnBackMode'),
    btnAdminLogin: RAG.qs('#btnAdminLogin'),
    username: RAG.qs('#username'),
    password: RAG.qs('#password'),
    err: RAG.qs('#loginError'),
    info: RAG.qs('#loginInfo')
  };

  const DEFAULT_ADMIN_TIP = '默认账号 admin；密码在 Configs/security.json 中仅保存 bcrypt 哈希。';

  /** 统一提示：只显示其中一种（error / info），传空字符串即隐藏 */
  function showMessage(node, message) {
    if (!node) return;
    node.textContent = message;
    node.hidden = !message;
  }

  function showError(message) {
    showMessage(els.err, message);
    if (message) showMessage(els.info, '');
  }

  function showInfo(message) {
    showMessage(els.info, message);
    if (message) showMessage(els.err, '');
  }

  /* ---------- 已登录（同一标签页内）直接进入主界面 ---------- */
  if (RAG.token.get()) {
    RAG.goToChat();
    return;
  }

  /* ---------- 展开管理员登录表单 ---------- */
  els.btnShowAdmin.addEventListener('click', function () {
    els.modeSelect.hidden = true;
    els.adminForm.hidden = false;
    showError('');
    showInfo(DEFAULT_ADMIN_TIP);
    setTimeout(function () { if (els.username) els.username.focus(); }, 60);
  });

  /* ---------- 返回模式选择 ---------- */
  els.btnBackMode.addEventListener('click', function () {
    els.adminForm.hidden = true;
    els.modeSelect.hidden = false;
    showError('');
    showInfo('');
    if (els.password) els.password.value = '';
  });

  /* ---------- 管理员登录 ---------- */
  els.adminForm.addEventListener('submit', async function (ev) {
    ev.preventDefault();
    const username = (els.username.value || '').trim();
    const password = els.password.value || '';
    if (!username || !password) {
      showError('请输入管理员账号与密码');
      return;
    }
    RAG.setBusy(els.btnAdminLogin, true, '登录中…');
    showError('');
    try {
      const data = await RAG.login(username, password);
      showInfo('✅ ' + (data.message || '登录成功') + '，正在进入主界面…');
      setTimeout(function () { RAG.goToChat(); }, 400);
    } catch (err) {
      // 401 → 账号或密码错误；其余为网络/服务异常
      showError('❌ ' + (err.status === 401 ? '账号或密码错误' : RAG.errText(err)));
      if (els.password) { els.password.value = ''; els.password.focus(); }
    } finally {
      RAG.setBusy(els.btnAdminLogin, false);
    }
  });

  /* ---------- 游客一键登录 ---------- */
  els.btnGuest.addEventListener('click', async function () {
    RAG.setBusy(els.btnGuest, true, '进入中…');
    showError('');
    try {
      const data = await RAG.guestLogin();
      showInfo('👤 ' + (data.message || '已进入') + '，正在跳转…');
      setTimeout(function () { RAG.goToChat(); }, 350);
    } catch (err) {
      showError('❌ ' + RAG.errText(err));
    } finally {
      RAG.setBusy(els.btnGuest, false);
    }
  });

  // 输入框聚焦时清掉错误提示，避免干扰
  [els.username, els.password].forEach(function (input) {
    if (input) input.addEventListener('input', function () { showError(''); });
  });
})();
