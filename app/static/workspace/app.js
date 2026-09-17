/* Bedrock Workspace client. Vanilla JS, no build step, no external
   dependencies — the app is served by this same FastAPI process so it
   works fully air-gapped. */
(() => {
  const API = "/workspace/api";
  let state = null;          // last /state response
  let view = "chats";        // chats | projects | memory | meetings | settings
  let currentChatId = null;
  let currentProjectId = null;
  let currentMeetingId = null;
  let composerAttachments = [];
  let pollHandle = null;
  let lastRenderSig = { center: "", activity: "", sidebarList: "" };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, props = {}, children = []) => {
    const node = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    });
    (Array.isArray(children) ? children : [children]).forEach((c) => {
      if (c == null) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  };
  const escapeHtml = (s) => (s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtTime = (iso) => { try { return new Date(iso).toLocaleString(); } catch { return iso || ""; } };
  const byId = (list, id) => (list || []).find((x) => x.id === id);

  async function api(path, opts = {}) {
    const res = await fetch(API + path, {
      method: opts.method || "GET",
      credentials: "same-origin",
      headers: opts.body ? { "Content-Type": "application/json" } : {},
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    if (res.status === 401) { showLogin(); throw new Error("signed out"); }
    if (!res.ok) {
      let message = "Something went wrong.";
      try { message = (await res.json()).detail || message; } catch {}
      throw new Error(message);
    }
    return res.status === 204 ? null : res.json();
  }

  // ---------------- login ----------------
  async function boot() {
    let cfg;
    try { cfg = await (await fetch(API + "/config")).json(); } catch { cfg = { demo: true }; }
    try {
      state = await api("/state");
      showApp();
      applyTheme();
      render();
      startPolling();
    } catch {
      showLogin(cfg);
    }
  }

  function showLogin(cfg) {
    stopPolling();
    $("#app").hidden = true;
    $("#login").hidden = false;
    if (!cfg) return;
    $("#demo-login-btn").hidden = !cfg.demo;
    $("#sso-login-link").hidden = !cfg.sso;
    $("#login-fallback").hidden = cfg.demo || cfg.sso;
    $("#login-copy").textContent = cfg.demo
      ? "This is a demo environment with fictional data."
      : "Sign in with your company account to continue.";
  }

  function showApp() {
    $("#login").hidden = true;
    $("#app").hidden = false;
  }

  $("#demo-login-btn").addEventListener("click", async () => {
    await fetch(API + "/demo-login", { method: "POST", credentials: "same-origin" });
    boot();
  });

  $("#logout-btn").addEventListener("click", async () => {
    await fetch(API + "/logout", { method: "POST", credentials: "same-origin" });
    state = null;
    showLogin({ demo: true });
  });

  function applyTheme() {
    const theme = state?.preferences?.[0]?.theme;
    if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  }

  // ---------------- polling ----------------
  function startPolling() {
    stopPolling();
    pollHandle = setInterval(async () => {
      if (document.hidden) return;
      try { state = await api("/state"); render(); } catch {}
    }, 1800);
  }
  function stopPolling() { if (pollHandle) clearInterval(pollHandle); pollHandle = null; }

  async function refresh() { state = await api("/state"); render(); }

  // ---------------- nav ----------------
  $$(".nav-item[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.classList.contains("disabled")) return;
      view = btn.dataset.view;
      currentProjectId = null;
      currentMeetingId = null;
      $$(".nav-item[data-view]").forEach((b) => b.classList.toggle("active", b === btn));
      document.getElementById("app").classList.remove("sidebar-open");
      render(true);
    });
  });

  // ---------------- mobile sidebar drawer ----------------
  $("#sidebar-menu-btn").addEventListener("click", () => appEl.classList.add("sidebar-open"));
  $("#sidebar-scrim").addEventListener("click", () => appEl.classList.remove("sidebar-open"));
  $("#activity-scrim").addEventListener("click", () => { appEl.classList.remove("activity-open"); appEl.classList.add("activity-collapsed"); });

  // ---------------- activity panel ----------------
  const appEl = $("#app");
  $("#activity-toggle").addEventListener("click", () => {
    appEl.classList.toggle("activity-open");
    appEl.classList.toggle("activity-collapsed");
  });
  $("#activity-collapse").addEventListener("click", () => {
    appEl.classList.add("activity-collapsed");
    appEl.classList.remove("activity-open");
  });
  $("#activity-popout").addEventListener("click", () => {
    const w = window.open("", "bedrock-agent-activity", "width=420,height=720");
    if (!w) return;
    w.document.write("<title>Agent Activity</title>");
    const link = w.document.createElement("link");
    link.rel = "stylesheet"; link.href = location.origin + "/static/workspace/styles.css";
    w.document.head.appendChild(link);
    w.document.body.style.margin = "0";
    w.document.body.appendChild(el("div", { id: "activity-body", style: "padding:12px" }));
    w.renderPopout = () => { w.document.getElementById("activity-body").innerHTML = $("#activity-body").innerHTML; };
    w.renderPopout();
    const iv = setInterval(() => { if (w.closed) return clearInterval(iv); w.renderPopout(); }, 1800);
  });

  // ---------------- render dispatch ----------------
  function render(force) {
    if (!state) return;
    $("#demo-badge").hidden = !state.config?.demo;
    $("#user-name").textContent = state.user?.name || "";
    renderSidebarList(force);
    renderCenter(force);
    renderActivity();
  }

  // ---------------- sidebar list per view ----------------
  function renderSidebarList(force) {
    const host = $("#sidebar-list");
    if (view === "chats") {
      const chats = [...state.chats].sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
      const sig = "chats:" + chats.map((c) => c.id + c.title + (c.pinned ? "p" : "")).join(",") + currentChatId;
      if (sig === lastRenderSig.sidebarList && !force) return;
      lastRenderSig.sidebarList = sig;
      host.innerHTML = "";
      host.appendChild(el("button", { class: "btn primary small", style: "margin-bottom:8px;width:100%", onclick: () => createChat() }, "+ New chat"));
      chats.forEach((c) => {
        const item = el("div", { class: "sidebar-list-item" + (c.id === currentChatId ? " active" : "") }, [
          el("div", { class: "title" }, (c.pinned ? "★ " : "") + (c.title || "New chat")),
          el("div", { class: "meta" }, c.projectId ? (byId(state.projects, c.projectId)?.name || "Project") : "Personal"),
        ]);
        item.addEventListener("click", () => { currentChatId = c.id; document.getElementById("app").classList.remove("sidebar-open"); render(true); });
        host.appendChild(item);
      });
    } else {
      lastRenderSig.sidebarList = "";
      host.innerHTML = "";
    }
  }

  // ---------------- center ----------------
  function renderCenter(force) {
    const title = { chats: "Chats", projects: "Projects", memory: "Memory", meetings: "Meetings", settings: "Settings" }[view] || "";
    $("#center-title").textContent = title;
    $("#center-actions").innerHTML = "";
    const body = $("#center-body");
    if (view === "chats") return renderChatView(body, force);
    if (view === "projects") return renderProjectsView(body);
    if (view === "memory") return renderMemoryView(body);
    if (view === "meetings") return renderMeetingsView(body);
    if (view === "settings") return renderSettingsView(body);
  }

  // ---- Chats ----
  function renderChatView(body, force) {
    const chat = currentChatId ? byId(state.chats, currentChatId) : null;
    const sig = "chat:" + (chat ? chat.id + (chat.messages || []).length + JSON.stringify(chat.artifactIds || []) : "none") + composerAttachments.join(",");
    if (sig === lastRenderSig.center && !force) return;
    lastRenderSig.center = sig;
    body.innerHTML = "";
    if (!chat) {
      body.appendChild(el("div", { class: "empty-state" }, [
        el("p", {}, "No chat selected."),
        el("button", { class: "btn primary", onclick: () => createChat() }, "Start a new chat"),
      ]));
      return;
    }
    const wrap = el("div", {});
    const thread = el("div", { class: "chat-thread" });
    (chat.messages || []).forEach((m) => {
      const bubble = el("div", { class: "msg " + (m.role === "user" ? "user" : "assistant") }, escapeHtml(m.text));
      if (m.role === "assistant") {
        const meta = [];
        if (m.demo) meta.push("Demo answer");
        if (m.citations && m.citations.length) meta.push(m.citations.length + " source(s) cited");
        if (meta.length) bubble.appendChild(el("div", { class: "msg-meta" }, meta.join(" · ")));
      }
      thread.appendChild(bubble);
    });
    if (!(chat.messages || []).length) {
      thread.appendChild(el("div", { class: "empty-state" }, "Attach a file or ask a question to begin."));
    }
    wrap.appendChild(thread);
    body.appendChild(wrap);
    body.scrollTop = body.scrollHeight;
    renderComposer(chat);
  }

  function renderComposer(chat) {
    let dock = document.getElementById("composer-wrap");
    if (dock) dock.remove();
    dock = el("div", { id: "composer-wrap" });
    const artifacts = (chat.artifactIds || []).map((id) => byId(state.artifacts, id)).filter(Boolean);
    if (artifacts.length) {
      const chips = el("div", { class: "attachment-chips" });
      artifacts.forEach((a) => {
        chips.appendChild(el("div", { class: "chip" }, [
          el("span", {}, a.name + (a.scanStatus === "review" ? " ⚠" : "")),
        ]));
      });
      dock.appendChild(chips);
    }
    const activeRun = state.runs.find((r) => r.chatId === chat.id && ["queued", "running", "waiting"].includes(r.status));
    const form = el("form", { id: "composer" });
    const modeSelect = el("select", { id: "mode-select" }, [
      el("option", { value: "auto" }, "Auto"),
      el("option", { value: "ask" }, "Ask"),
      el("option", { value: "research" }, "Research"),
      el("option", { value: "analyze" }, "Analyze"),
      el("option", { value: "agent" }, "Agent"),
    ]);
    const fileInput = el("input", { type: "file", hidden: "hidden" });
    fileInput.addEventListener("change", () => { if (fileInput.files[0]) attachFile(chat, fileInput.files[0]); });
    const attachBtn = el("button", { type: "button", class: "icon-btn", title: "Attach a file", onclick: () => fileInput.click() }, "📎");
    const textarea = el("textarea", { placeholder: activeRun ? "A task is already running for this chat…" : "Ask a question…", rows: "1" });
    if (activeRun) textarea.disabled = true;
    const sendBtn = el("button", { class: "btn primary", type: "submit" }, activeRun ? "Running…" : "Send");
    if (activeRun) sendBtn.disabled = true;
    form.appendChild(fileInput);
    form.appendChild(attachBtn);
    form.appendChild(textarea);
    form.appendChild(modeSelect);
    form.appendChild(sendBtn);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = textarea.value.trim();
      if (!text) return;
      textarea.value = ""; textarea.disabled = true; sendBtn.disabled = true; sendBtn.textContent = "Sending…";
      try {
        await api("/runs", { method: "POST", body: { chatId: chat.id, prompt: text, mode: modeSelect.value } });
        await refresh();
      } catch (err) {
        alert(err.message);
        await refresh();
      }
    });
    dock.appendChild(form);
    $("#center").appendChild(dock);
  }

  async function createChat(projectId = null) {
    const chat = await api("/records/chats", { method: "POST", body: { data: { title: "New chat", projectId, messages: [], artifactIds: [], pinned: false, archived: false } } });
    currentChatId = chat.id;
    view = "chats";
    $$(".nav-item[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === "chats"));
    document.getElementById("app").classList.remove("sidebar-open");
    await refresh();
  }

  async function attachFile(chat, file) {
    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = reader.result.split(",")[1];
      try {
        await api("/uploads", { method: "POST", body: { name: file.name, base64, chatId: chat.id, projectId: chat.projectId || null } });
        await refresh();
      } catch (err) { alert(err.message); }
    };
    reader.readAsDataURL(file);
  }

  // ---- Projects ----
  function renderProjectsView(body) {
    body.innerHTML = "";
    const pane = el("div", { class: "pane" });
    if (currentProjectId) {
      const project = byId(state.projects, currentProjectId);
      if (!project) { currentProjectId = null; return renderProjectsView(body); }
      const back = el("a", { class: "back-link", onclick: () => { currentProjectId = null; renderCenter(true); } }, "← All projects");
      pane.appendChild(back);
      const card = el("div", { class: "detail-card" });
      card.appendChild(el("h2", {}, project.name));
      card.appendChild(el("p", { class: "muted" }, project.description || ""));
      card.appendChild(el("div", { class: "section-title" }, "Members"));
      card.appendChild(el("ul", { class: "kv-list" }, (project.members || []).map((m) => el("li", {}, `${m.name} — ${m.role}`))));
      card.appendChild(el("button", { class: "btn primary small", style: "margin-top:12px", onclick: () => createChat(project.id) }, "Start a chat in this project"));
      pane.appendChild(card);
      const memory = state.memory.filter((m) => m.projectId === project.id);
      pane.appendChild(el("div", { class: "section-title" }, "Project memory"));
      pane.appendChild(el("ul", { class: "kv-list" }, memory.length ? memory.map((m) => el("li", {}, m.title)) : [el("li", { class: "muted" }, "No project memory yet.")]));
      const meetings = state.meetings.filter((m) => m.projectId === project.id);
      pane.appendChild(el("div", { class: "section-title" }, "Meetings"));
      pane.appendChild(el("ul", { class: "kv-list" }, meetings.length ? meetings.map((m) => el("li", {}, m.title)) : [el("li", { class: "muted" }, "No meetings yet.")]));
      body.appendChild(pane);
      return;
    }
    pane.appendChild(el("div", { class: "pane-header" }, [
      el("span", { class: "muted" }, state.projects.length + " project(s)"),
      el("button", { class: "btn primary small", onclick: () => openCreateForm("project") }, "+ New project"),
    ]));
    const grid = el("div", { class: "card-grid" });
    state.projects.forEach((p) => {
      const c = el("div", { class: "card" }, [
        el("h3", {}, p.name),
        el("p", {}, p.description || ""),
        el("span", { class: "badge pill" }, (p.members || []).length + " member(s)"),
      ]);
      c.addEventListener("click", () => { currentProjectId = p.id; renderCenter(true); });
      grid.appendChild(c);
    });
    pane.appendChild(grid);
    body.appendChild(pane);
  }

  // ---- Memory ----
  function renderMemoryView(body) {
    body.innerHTML = "";
    const pane = el("div", { class: "pane" });
    pane.appendChild(el("div", { class: "pane-header" }, [
      el("span", { class: "muted" }, "Personal, project and team context the assistant is allowed to use."),
      el("button", { class: "btn primary small", onclick: () => openCreateForm("memory") }, "+ New memory"),
    ]));
    ["Personal", "Project", "Team", "Company"].forEach((scope) => {
      const items = state.memory.filter((m) => (m.scope || "Personal") === scope);
      if (!items.length) return;
      pane.appendChild(el("div", { class: "section-title" }, scope));
      const grid = el("div", { class: "card-grid" });
      items.forEach((m) => {
        grid.appendChild(el("div", { class: "card", style: "cursor:default" }, [
          el("h3", {}, m.title),
          el("p", {}, (m.content || "").slice(0, 140)),
        ]));
      });
      pane.appendChild(grid);
    });
    if (!state.memory.length) pane.appendChild(el("p", { class: "muted" }, "No memory saved yet."));
    body.appendChild(pane);
  }

  // ---- Meetings ----
  function renderMeetingsView(body) {
    body.innerHTML = "";
    const pane = el("div", { class: "pane" });
    if (currentMeetingId) {
      const meeting = byId(state.meetings, currentMeetingId);
      if (!meeting) { currentMeetingId = null; return renderMeetingsView(body); }
      pane.appendChild(el("a", { class: "back-link", onclick: () => { currentMeetingId = null; renderCenter(true); } }, "← All meetings"));
      const card = el("div", { class: "detail-card" });
      card.appendChild(el("h2", {}, meeting.title));
      card.appendChild(el("p", { class: "muted" }, meeting.summary || ""));
      const block = (label, items) => el("div", { class: "list-block" }, [
        el("div", { class: "section-title" }, label),
        el("ul", {}, (items || []).length ? items.map((i) => el("li", {}, i)) : [el("li", { class: "muted" }, "None extracted.")]),
      ]);
      card.appendChild(block("Decisions", meeting.decisions));
      card.appendChild(block("Action items", meeting.actions));
      card.appendChild(block("Deadlines", meeting.deadlines));
      card.appendChild(block("Open questions", meeting.questions));
      card.appendChild(block("Findings", meeting.findings));
      card.appendChild(block("People", meeting.people));
      if (meeting.transcript) {
        card.appendChild(el("div", { class: "section-title" }, "Transcript"));
        card.appendChild(el("div", { class: "transcript-box" }, meeting.transcript));
      }
      card.appendChild(el("p", { class: "muted small", style: "margin-top:10px" }, "Rule-based extraction — review before treating as verified minutes."));
      pane.appendChild(card);
      body.appendChild(pane);
      return;
    }
    pane.appendChild(el("div", { class: "pane-header" }, [
      el("span", { class: "muted" }, state.meetings.length + " meeting(s)"),
    ]));
    const grid = el("div", { class: "card-grid" });
    state.meetings.forEach((m) => {
      const c = el("div", { class: "card" }, [
        el("h3", {}, m.title),
        el("p", {}, m.summary || ""),
      ]);
      c.addEventListener("click", () => { currentMeetingId = m.id; renderCenter(true); });
      grid.appendChild(c);
    });
    pane.appendChild(grid);
    if (!state.meetings.length) pane.appendChild(el("p", { class: "muted" }, "No meetings yet. Attach a transcript file from a chat, or import one below."));
    body.appendChild(pane);
  }

  // ---- Settings ----
  function renderSettingsView(body) {
    body.innerHTML = "";
    const pane = el("div", { class: "pane" });
    const cfg = state.config || {};
    pane.appendChild(el("div", { class: "section-title" }, "Environment"));
    pane.appendChild(el("div", { class: "settings-grid" }, [
      el("div", { class: "card" }, [el("h3", {}, "Mode"), el("p", {}, cfg.demo ? "Demo (fictional data)" : "Real")]),
      el("div", { class: "card" }, [el("h3", {}, "AI provider"), el("p", {}, cfg.ai || "Unknown")]),
      el("div", { class: "card" }, [el("h3", {}, "Authentication"), el("p", {}, cfg.auth || "Unknown")]),
    ]));
    pane.appendChild(el("div", { class: "section-title" }, "Company knowledge connectors"));
    pane.appendChild(el("div", { class: "settings-grid" }, (cfg.connectors || []).map((c) => el("div", { class: "card" }, [el("h3", {}, c.name), el("p", {}, c.status)]))));
    const prefs = (state.preferences || [])[0] || {};
    pane.appendChild(el("div", { class: "section-title" }, "Preferences"));
    const card = el("div", { class: "detail-card" });
    card.appendChild(el("p", {}, [el("b", {}, "Theme: "), prefs.theme || "light"]));
    card.appendChild(el("p", {}, [el("b", {}, "Buddy assistant: "), prefs.buddy ? "On" : "Off"]));
    card.appendChild(el("p", {}, [el("b", {}, "Data retention: "), prefs.retention || "Not set"]));
    card.appendChild(el("p", {}, [el("b", {}, "Scientific vocabulary: "), el("br"), el("span", { class: "muted" }, (prefs.vocabulary || "").split("\n").join(", "))]));
    pane.appendChild(card);
    body.appendChild(pane);
  }

  // ---- inline create forms (memory / project) ----
  function openCreateForm(kind) {
    const overlay = el("div", { class: "login-screen", style: "position:fixed;inset:0;z-index:30;background:rgba(0,0,0,0.35)" });
    const card = el("div", { class: "login-card", style: "text-align:left;width:380px" });
    const form = el("form", { class: "simple-form" });
    if (kind === "project") {
      form.appendChild(el("div", {}, [el("label", {}, "Name"), el("input", { name: "name", required: "required" })]));
      form.appendChild(el("div", {}, [el("label", {}, "Description"), el("textarea", { name: "description" })]));
    } else {
      form.appendChild(el("div", {}, [el("label", {}, "Title"), el("input", { name: "title", required: "required" })]));
      form.appendChild(el("div", {}, [el("label", {}, "Content"), el("textarea", { name: "content", required: "required" })]));
      const scopeSelect = el("select", { name: "scope" }, [el("option", {}, "Personal"), el("option", {}, "Project")]);
      form.appendChild(el("div", {}, [el("label", {}, "Scope"), scopeSelect]));
      const projSelect = el("select", { name: "projectId" }, state.projects.map((p) => el("option", { value: p.id }, p.name)));
      form.appendChild(el("div", {}, [el("label", {}, "Project (if scope is Project)"), projSelect]));
    }
    const actions = el("div", { style: "display:flex;gap:8px;justify-content:flex-end;margin-top:6px" });
    const cancelBtn = el("button", { type: "button", class: "btn ghost small" }, "Cancel");
    const submitBtn = el("button", { type: "submit", class: "btn primary small" }, "Create");
    actions.appendChild(cancelBtn); actions.appendChild(submitBtn);
    form.appendChild(actions);
    cancelBtn.addEventListener("click", () => overlay.remove());
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fd = new FormData(form);
      try {
        if (kind === "project") {
          await api("/records/projects", { method: "POST", body: { data: { name: fd.get("name"), description: fd.get("description"), members: [{ name: state.user.name, role: "Owner" }], scope: "Project" } } });
        } else {
          const scope = fd.get("scope");
          await api("/records/memory", { method: "POST", body: { data: { title: fd.get("title"), content: fd.get("content"), scope, projectId: scope === "Project" ? fd.get("projectId") : null } } });
        }
        overlay.remove();
        await refresh();
      } catch (err) { alert(err.message); }
    });
    card.appendChild(el("h2", { style: "margin-top:0" }, kind === "project" ? "New project" : "New memory"));
    card.appendChild(form);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
  }

  // ---------------- activity panel ----------------
  function renderActivity() {
    const runs = [...state.runs].sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
    const sig = JSON.stringify(runs.map((r) => [r.id, r.status, r.currentStep, (r.steps || []).map((s) => s.status)]));
    if (sig === lastRenderSig.activity) return;
    lastRenderSig.activity = sig;
    const host = $("#activity-body");
    host.innerHTML = "";
    if (!runs.length) { host.appendChild(el("div", { class: "empty-state" }, "No agent activity yet. Send a message to start one.")); return; }
    runs.forEach((r) => {
      const chat = byId(state.chats, r.chatId);
      const card = el("div", { class: "run-card" });
      card.appendChild(el("div", { class: "run-title" }, r.title || r.prompt));
      card.appendChild(el("div", { class: "run-meta" }, [
        el("span", { class: "badge status-" + r.status }, r.status),
        el("span", {}, r.role || ""),
      ]));
      if (r.steps && r.steps.length) {
        const list = el("div", { class: "run-step-list" });
        r.steps.forEach((s) => list.appendChild(el("div", { class: "run-step " + s.status }, [el("span", { class: "dot" }), el("span", {}, s.title)])));
        card.appendChild(list);
      } else if (["queued", "running"].includes(r.status)) {
        card.appendChild(el("div", { class: "run-current-step" }, [el("span", { class: "dot" }), el("span", {}, r.currentStep || "Working…")]));
      }
      if (r.status === "waiting") {
        const actions = el("div", { class: "run-actions" });
        actions.appendChild(el("button", { class: "btn primary small", onclick: () => decide(r.id, "approve") }, "Approve"));
        actions.appendChild(el("button", { class: "btn danger small", onclick: () => decide(r.id, "reject") }, "Reject"));
        card.appendChild(actions);
        if (r.preview) card.appendChild(el("p", { class: "small muted" }, r.preview));
      } else if (["queued", "running"].includes(r.status)) {
        card.appendChild(el("div", { class: "run-actions" }, [el("button", { class: "btn ghost small", onclick: () => cancelRun(r.id) }, "Cancel")]));
      } else if (r.status === "failed" && r.error) {
        card.appendChild(el("p", { class: "small", style: "color:var(--bad)" }, r.error));
      }
      if (chat) card.appendChild(el("a", { class: "run-answer-link", onclick: () => {
        view = "chats"; currentChatId = chat.id;
        $$(".nav-item[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === "chats"));
        appEl.classList.remove("activity-open"); appEl.classList.add("activity-collapsed");
        render(true);
      } }, "View in chat →"));
      host.appendChild(card);
    });
  }

  async function decide(runId, decision) {
    try { await api(`/runs/${runId}/decision`, { method: "POST", body: { decision } }); await refresh(); }
    catch (err) { alert(err.message); }
  }
  async function cancelRun(runId) {
    try { await api(`/runs/${runId}/cancel`, { method: "POST" }); await refresh(); }
    catch (err) { alert(err.message); }
  }

  boot();
})();
