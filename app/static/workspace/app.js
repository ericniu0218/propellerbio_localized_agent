/* Bedrock Workspace client. Vanilla JS, no build step, no external
   dependencies — the app is served by this same FastAPI process so it
   works fully air-gapped. */
(() => {
  const API = "/workspace/api";
  let state = null;
  let view = "chats";
  let currentChatId = null;
  let currentProjectId = null;
  let currentMeetingId = null;
  let pollHandle = null;
  let lastRenderSig = { center: "", activity: "", sidebarList: "" };

  // Live recording state. Held outside `state` because it is this browser tab's
  // business only — nothing about a half-finished recording belongs on the
  // server until the audio is uploaded.
  let recording = null;          // the active BedrockRecorder.Session
  let liveSessionId = null;      // preview session on the server, if any
  let livePreviewText = "";
  let recordingJobs = [];        // uploads being transcribed, newest first
  let recordingJobPoll = null;

  function getEnterToSend() {
    try { return localStorage.getItem("bedrock_enter_to_send") !== "false"; } catch { return true; }
  }
  function setEnterToSend(v) {
    try { localStorage.setItem("bedrock_enter_to_send", v ? "true" : "false"); } catch {}
  }

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, props = {}, children = []) => {
    const node = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
      if (v == null || v === false) return;
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    });
    (Array.isArray(children) ? children : [children]).forEach((c) => {
      if (c == null) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  };
  const byId = (list, id) => (list || []).find((x) => x.id === id);
  const initials = (name) => (name || "?").trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
  const fmtSize = (n) => (n > 1e6 ? (n / 1e6).toFixed(1) + " MB" : n > 1e3 ? (n / 1e3).toFixed(1) + " KB" : n + " B");

  // ---------------- icons ----------------
  const ICON_PATHS = {
    menu: '<path d="M3 6h18M3 12h18M3 18h18"/>',
    chat: '<path d="M21 11.5a8.4 8.4 0 0 1-8.4 8.4 8.3 8.3 0 0 1-3.8-.9L3 21l1.9-5.8a8.3 8.3 0 0 1-.9-3.7A8.4 8.4 0 0 1 12.5 3a8.4 8.4 0 0 1 8.4 8.4z"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/>',
    brain: '<path d="M6 3h12v18l-6-4-6 4V3z"/>',
    calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    tool: '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.8 2.8-2-2 2.8-2.8z"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6V21a2 2 0 1 1-4 0v-.2a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.6-1H3a2 2 0 1 1 0-4h.2a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H9a1.7 1.7 0 0 0 1-1.6V3a2 2 0 1 1 4 0v.2a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V9a1.7 1.7 0 0 0 1.6 1H21a2 2 0 1 1 0 4h-.2a1.7 1.7 0 0 0-1.6 1z"/>',
    agent: '<rect x="4" y="8" width="16" height="11" rx="3"/><circle cx="9" cy="13.5" r="1.1" fill="currentColor" stroke="none"/><circle cx="15" cy="13.5" r="1.1" fill="currentColor" stroke="none"/><path d="M12 8V4M9 4h6"/>',
    popout: '<path d="M14 4h6v6M20 4 10 14M8 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    close: '<path d="M6 6l12 12M18 6 6 18"/>',
    paperclip: '<path d="M21 11.5 12.5 20a4.5 4.5 0 0 1-6.4-6.4L14.6 5a3 3 0 0 1 4.2 4.2l-8.5 8.5a1.5 1.5 0 0 1-2.1-2.1l7.1-7.1"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    chevrondown: '<path d="M6 9l6 6 6-6"/>',
    chevronright: '<path d="M9 6l6 6-6 6"/>',
    trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13"/>',
    send: '<path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z"/>',
    upload: '<path d="M12 16V4M7 9l5-5 5 5M4 20h16"/>',
    download: '<path d="M12 4v12M7 11l5 5 5-5M4 20h16"/>',
  };
  const icon = (name, cls = "") => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("class", "icon " + cls);
    svg.innerHTML = ICON_PATHS[name] || "";
    return svg;
  };
  function paintIcons(root = document) {
    $$("[data-icon]", root).forEach((node) => {
      if (node.dataset.iconDone) return;
      const name = node.dataset.icon;
      node.insertBefore(icon(name), node.firstChild);
      node.dataset.iconDone = "1";
    });
  }

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
    paintIcons();
    let cfg;
    try { cfg = await (await fetch(API + "/config")).json(); } catch { cfg = { demo: true }; }
    try {
      state = await api("/state");
      showApp();
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
    $("#dev-login-form").hidden = !cfg.devLogin;
    $("#login-fallback").hidden = cfg.demo || cfg.sso || cfg.devLogin;
    $("#login-copy").textContent = cfg.demo
      ? "This is a demo environment with fictional data."
      : cfg.devLogin
      ? "No company login is configured yet — enter a name to continue."
      : "Sign in with your company account to continue.";
  }
  function showApp() { $("#login").hidden = true; $("#app").hidden = false; }

  $("#demo-login-btn").addEventListener("click", async () => {
    await fetch(API + "/demo-login", { method: "POST", credentials: "same-origin" });
    boot();
  });
  $("#dev-login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("#dev-login-name").value.trim();
    if (!name) return;
    await api("/dev-login", { method: "POST", body: { name } });
    boot();
  });
  $("#logout-btn").addEventListener("click", async () => {
    await fetch(API + "/logout", { method: "POST", credentials: "same-origin" });
    state = null;
    showLogin({ demo: true });
  });

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
  // A backgrounded tab has its poll interval skipped (line above) and is
  // additionally throttled by the browser itself, so a run can finish while
  // hidden and sit stale for a long time after the tab is shown again.
  // Catch up immediately instead of waiting for the next tick.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state) refresh().catch(() => {});
  });

  // ---------------- nav ----------------
  $$(".nav-item[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.classList.contains("disabled")) return;
      view = btn.dataset.view;
      currentProjectId = null;
      currentMeetingId = null;
      $$(".nav-item[data-view]").forEach((b) => b.classList.toggle("active", b === btn));
      appEl.classList.remove("sidebar-open");
      render(true);
    });
  });

  // ---------------- mobile sidebar drawer ----------------
  const appEl = $("#app");
  $("#sidebar-menu-btn").addEventListener("click", () => appEl.classList.add("sidebar-open"));
  $("#sidebar-scrim").addEventListener("click", () => appEl.classList.remove("sidebar-open"));

  // ---------------- activity drawer ----------------
  function openActivity() { appEl.classList.add("activity-open"); }
  function closeActivity() { appEl.classList.remove("activity-open"); }
  $("#activity-toggle").addEventListener("click", () => appEl.classList.toggle("activity-open"));
  $("#activity-collapse").addEventListener("click", closeActivity);
  $("#activity-popout").addEventListener("click", () => {
    const w = window.open("about:blank", "bedrock-agent-activity", "width=420,height=720");
    if (!w) {
      alert("Your browser blocked this pop-up. Allow pop-ups for this page and click Pop out again.");
      return;
    }
    w.document.title = "Agent Activity";
    const link = w.document.createElement("link");
    link.rel = "stylesheet"; link.href = location.origin + "/static/workspace/styles.css";
    w.document.head.appendChild(link);
    w.document.body.style.cssText = "margin:0;background:#171614;color:#ece6dc";
    const note = w.document.createElement("div");
    note.style.cssText = "padding:10px 14px;font-size:11px;color:#8c8478;border-bottom:1px solid #37332e";
    note.textContent = "Live view — approve, reject, and cancel from the main window.";
    w.document.body.appendChild(note);
    const container = w.document.createElement("div");
    container.id = "activity-body";
    container.style.padding = "14px";
    w.document.body.appendChild(container);
    const paint = () => { container.innerHTML = $("#activity-body").innerHTML; };
    paint();
    const iv = setInterval(() => { if (w.closed) { clearInterval(iv); return; } paint(); }, 1500);
  });

  // resizable drawer
  (() => {
    const handle = $("#activity-resizer");
    let dragging = false;
    handle.addEventListener("mousedown", (e) => {
      dragging = true; handle.classList.add("dragging");
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const w = Math.min(720, Math.max(300, window.innerWidth - e.clientX));
      document.documentElement.style.setProperty("--activity-w", w + "px");
    });
    window.addEventListener("mouseup", () => { dragging = false; handle.classList.remove("dragging"); });
  })();

  // ---------------- modal ----------------
  function openModal({ title, sub, body, wide, footer }) {
    const root = $("#modal-root");
    root.innerHTML = "";
    root.hidden = false;
    const backdrop = el("div", { class: "modal-backdrop", onclick: close });
    const modal = el("div", { class: "modal" + (wide ? " wide" : "") });
    modal.appendChild(el("button", { class: "icon-btn modal-close", onclick: close }, icon("close")));
    modal.appendChild(el("h2", {}, title));
    if (sub) modal.appendChild(el("p", { class: "modal-sub" }, sub));
    modal.appendChild(body);
    if (footer) modal.appendChild(el("div", { class: "modal-footer" }, footer));
    root.appendChild(backdrop);
    root.appendChild(modal);
    function close() { root.hidden = true; root.innerHTML = ""; }
    return { close, modal };
  }

  // ---------------- render dispatch ----------------
  function render(force) {
    if (!state) return;
    const user = state.user || {};
    $("#user-name").textContent = user.name || "";
    $("#user-avatar").textContent = initials(user.name);
    renderSidebarList(force);
    renderCenter(force);
    renderActivity();
  }

  function renderSidebarList(force) {
    const host = $("#sidebar-list");
    if (view === "chats") {
      const chats = [...state.chats].sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
      const sig = "chats:" + chats.map((c) => c.id + c.title + (c.pinned ? "p" : "")).join(",") + currentChatId;
      if (sig === lastRenderSig.sidebarList && !force) return;
      lastRenderSig.sidebarList = sig;
      host.innerHTML = "";
      const newBtn = el("button", { class: "btn primary small", style: "margin-bottom:10px;width:100%;justify-content:center", onclick: () => createChat() }, [icon("plus"), "New chat"]);
      host.appendChild(newBtn);
      chats.forEach((c) => {
        const info = el("div", { class: "sidebar-list-item-info" }, [
          el("div", { class: "title" }, (c.pinned ? "★ " : "") + (c.title || "New chat")),
          el("div", { class: "meta" }, c.projectId ? (byId(state.projects, c.projectId)?.name || "Project") : "Personal"),
        ]);
        const forgetBtn = el("button", { class: "icon-btn small", title: "Forget this chat", onclick: async (e) => {
          e.stopPropagation();
          try {
            await api(`/records/chats/${c.id}`, { method: "DELETE" });
            if (currentChatId === c.id) currentChatId = null;
            await refresh();
          } catch (err) { alert(err.message); }
        } }, icon("trash"));
        const item = el("div", { class: "sidebar-list-item" + (c.id === currentChatId ? " active" : "") }, [info, forgetBtn]);
        item.addEventListener("click", () => { currentChatId = c.id; appEl.classList.remove("sidebar-open"); render(true); });
        host.appendChild(item);
      });
    } else {
      lastRenderSig.sidebarList = "";
      host.innerHTML = "";
    }
  }

  function renderCenter(force) {
    const title = { chats: "Chats", projects: "Projects", memory: "Memory", meetings: "Meetings", settings: "Settings" }[view] || "";
    $("#center-title").textContent = title;
    $("#center-actions").innerHTML = "";
    const body = $("#center-body");
    if (view === "chats") return renderChatView(body, force);
    if (view === "projects") return renderProjectsView(body);
    if (view === "memory") return renderMemoryView(body);
    if (view === "meetings") return renderMeetingsView(body, force);
    if (view === "settings") return renderSettingsView(body);
  }

  // ==================== CHATS ====================
  function renderChatView(body, force) {
    const chat = currentChatId ? byId(state.chats, currentChatId) : null;
    const isPending = (r) => ["queued", "running", "waiting"].includes(r.status);
    const activeAskRun = chat && state.runs.find((r) => r.chatId === chat.id && ["ask", "auto"].includes(r.mode) && isPending(r));
    const activeAgentRun = chat && state.runs.find((r) => r.chatId === chat.id && !["ask", "auto"].includes(r.mode) && isPending(r));
    const sig = "chat:" + (chat ? chat.id + (chat.messages || []).length + JSON.stringify(chat.artifactIds || [])
      + (activeAskRun ? activeAskRun.id + activeAskRun.status : "") + (activeAgentRun ? activeAgentRun.id + activeAgentRun.status : "") : "none");
    if (sig === lastRenderSig.center && !force) return;
    lastRenderSig.center = sig;
    body.innerHTML = "";
    if (!chat) {
      body.appendChild(el("div", { class: "empty-state" }, [
        icon("chat"),
        el("p", {}, "No chat selected."),
        el("button", { class: "btn primary", onclick: () => createChat() }, "Start a new chat"),
      ]));
      const dock = document.getElementById("composer-wrap"); if (dock) dock.remove();
      return;
    }
    const thread = el("div", { class: "chat-thread" });
    (chat.messages || []).forEach((m) => {
      if (m.role === "user") {
        thread.appendChild(el("div", { class: "msg user" }, m.text));
      } else {
        const meta = [];
        if (m.citations && m.citations.length) meta.push(m.citations.length + " source(s) cited");
        thread.appendChild(el("div", { class: "msg assistant" }, [
          el("span", { class: "avatar" }, "AI"),
          el("div", { class: "bubble" }, [
            el("div", {}, m.text),
            meta.length ? el("div", { class: "msg-meta" }, meta.join(" · ")) : null,
          ]),
        ]));
      }
    });
    if (activeAskRun) {
      thread.appendChild(el("div", { class: "thinking-row" }, [
        el("span", { class: "avatar" }, "AI"),
        el("div", { class: "thinking-dots" }, [el("span"), el("span"), el("span")]),
      ]));
    }
    if (!(chat.messages || []).length && !activeAskRun) {
      thread.appendChild(el("div", { class: "empty-state" }, "Attach a file or ask a question to begin."));
    }
    body.appendChild(thread);
    body.scrollTop = body.scrollHeight;
    renderComposer(chat, activeAskRun, activeAgentRun);
  }

  function renderComposer(chat, activeAskRun, activeAgentRun) {
    let dock = document.getElementById("composer-wrap");
    if (dock) dock.remove();
    dock = el("div", { id: "composer-wrap" });
    const artifacts = (chat.artifactIds || []).map((id) => byId(state.artifacts, id)).filter(Boolean);
    if (artifacts.length) {
      const chips = el("div", { class: "attachment-chips" });
      artifacts.forEach((a) => chips.appendChild(el("div", { class: "chip" }, a.name + (a.scanStatus === "review" ? " ⚠" : ""))));
      dock.appendChild(chips);
    }
    const form = el("form", { id: "composer" });
    const fileInput = el("input", { type: "file", hidden: "hidden" });
    fileInput.addEventListener("change", () => { if (fileInput.files[0]) attachFile(chat, fileInput.files[0]); });
    const attachBtn = el("button", { type: "button", class: "icon-btn", title: "Attach a file", onclick: () => fileInput.click() }, icon("paperclip"));
    const textarea = el("textarea", { placeholder: activeAskRun ? "Waiting for a reply…" : "Ask a question…", rows: "1" });
    textarea.disabled = !!activeAskRun;
    textarea.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" || e.shiftKey) return;
      const enterSends = getEnterToSend();
      if (enterSends) { e.preventDefault(); form.requestSubmit(); }
      // else: fall through to the browser's default newline insertion.
    });
    const sendBtn = el("button", { class: "btn primary small", type: "submit", disabled: activeAskRun ? "disabled" : null }, [icon("send"), "Send"]);
    const agentBtn = el("button", { type: "button", class: "btn small", disabled: activeAgentRun ? "disabled" : null,
      title: activeAgentRun ? "An agent is already working on this chat" : "",
      onclick: () => openAgentPicker(chat, textarea.value) }, [icon("agent"), activeAgentRun ? "Agent running…" : "Create Agent"]);
    form.appendChild(fileInput);
    form.appendChild(attachBtn);
    form.appendChild(textarea);
    form.appendChild(agentBtn);
    form.appendChild(sendBtn);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = textarea.value.trim();
      if (!text) return;
      textarea.value = ""; textarea.disabled = true; sendBtn.disabled = true;
      try {
        await api("/runs", { method: "POST", body: { chatId: chat.id, prompt: text, mode: "ask" } });
        await refresh();
      } catch (err) { alert(err.message); await refresh(); }
    });
    dock.appendChild(form);

    const enterHint = el("div", { class: "composer-hint" });
    const enterSwitch = el("label", { class: "switch small" }, [el("input", { type: "checkbox", checked: getEnterToSend() ? "checked" : null }), el("span", { class: "track" })]);
    enterSwitch.querySelector("input").addEventListener("change", (e) => { setEnterToSend(e.target.checked); paintHint(); });
    function paintHint() { hintLabel.textContent = getEnterToSend() ? "Enter sends · Shift+Enter for a new line" : "Enter adds a new line"; }
    const hintLabel = el("span", {});
    paintHint();
    enterHint.appendChild(enterSwitch);
    enterHint.appendChild(hintLabel);
    dock.appendChild(enterHint);

    $("#center").appendChild(dock);
  }

  async function createChat(projectId = null) {
    const chat = await api("/records/chats", { method: "POST", body: { data: { title: "New chat", projectId, messages: [], artifactIds: [], pinned: false, archived: false } } });
    currentChatId = chat.id;
    view = "chats";
    $$(".nav-item[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === "chats"));
    appEl.classList.remove("sidebar-open");
    await refresh();
  }

  async function attachFile(chat, file, projectId) {
    const reader = new FileReader();
    return new Promise((resolve, reject) => {
      reader.onload = async () => {
        const base64 = reader.result.split(",")[1];
        try {
          const artifact = await api("/uploads", { method: "POST", body: { name: file.name, base64, chatId: chat ? chat.id : null, projectId: projectId || (chat && chat.projectId) || null } });
          await refresh();
          resolve(artifact);
        } catch (err) { alert(err.message); reject(err); }
      };
      reader.readAsDataURL(file);
    });
  }

  // ---- Agent picker modal ----
  const AGENTS = [
    { mode: "research", name: "Research", tagline: "Plans a broader look across your sources before answering.",
      detail: "A Director agent breaks the question into steps and coordinates sub-agents across your project's sources. Best for open-ended or multi-part questions that need more than one pass to answer well." },
    { mode: "analyze", name: "Analyze", tagline: "Deep technical read of the files attached to this chat.",
      detail: "A Specialist agent works through attached datasets in detail — statistics, trends, and flags. Best once you have a file attached and want a rigorous, structured read of it." },
    { mode: "agent", name: "Agent", tagline: "Autonomous multi-step agent that can use tools.",
      detail: "Runs a full agent loop that can take several actions on your behalf. It will pause and ask for your approval before anything high-stakes. Best for multi-step tasks, not simple questions." },
  ];
  function openAgentPicker(chat, prefill) {
    let selected = AGENTS[0].mode;
    let expanded = null;
    const cardsHost = el("div", { class: "agent-cards" });
    const task = el("textarea", { placeholder: "What should the agent do?", rows: "3" });
    task.value = prefill || "";
    function paintCards() {
      cardsHost.innerHTML = "";
      AGENTS.forEach((a) => {
        const isSel = a.mode === selected;
        const card = el("div", { class: "agent-card" + (isSel ? " selected" : "") });
        const head = el("div", { class: "agent-card-head" }, [
          el("div", { class: "agent-icon" }, icon("agent")),
          el("div", {}, [el("div", { class: "agent-card-title" }, a.name), el("div", { class: "agent-card-tagline" }, a.tagline)]),
          el("button", { type: "button", class: "icon-btn agent-card-expand", onclick: (e) => { e.stopPropagation(); expanded = expanded === a.mode ? null : a.mode; paintCards(); } }, icon(expanded === a.mode ? "chevrondown" : "chevronright")),
        ]);
        card.appendChild(head);
        if (expanded === a.mode) card.appendChild(el("div", { class: "agent-card-detail" }, a.detail));
        card.addEventListener("click", () => { selected = a.mode; paintCards(); });
        cardsHost.appendChild(card);
      });
    }
    paintCards();
    const body = el("div", { class: "simple-form" }, [
      cardsHost,
      el("div", { class: "field" }, [el("label", {}, "Task"), task]),
    ]);
    const startBtn = el("button", { class: "btn primary" }, "Start agent");
    const { close } = openModal({
      title: "Start an agent", sub: "Choose the agent best suited to the task, then describe what it should do.",
      body, wide: true,
      footer: [el("button", { class: "btn ghost" }, "Cancel"), startBtn],
    });
    $$("#modal-root .modal-footer .btn.ghost").forEach((b) => b.addEventListener("click", close));
    startBtn.addEventListener("click", async () => {
      const prompt = task.value.trim();
      if (!prompt) return;
      startBtn.disabled = true; startBtn.textContent = "Starting…";
      try {
        await api("/runs", { method: "POST", body: { chatId: chat.id, prompt, mode: selected } });
        close();
        openActivity();
        await refresh();
      } catch (err) { alert(err.message); startBtn.disabled = false; startBtn.textContent = "Start agent"; }
    });
  }

  // ==================== PROJECTS ====================
  function renderProjectsView(body) {
    body.innerHTML = "";
    const pane = el("div", { class: "pane" });
    if (currentProjectId) return renderProjectDetail(pane, body);
    pane.appendChild(el("p", { class: "pane-intro" }, "Shared spaces for a program's chats, memory, meetings and files."));
    if (!state.projects.length) {
      pane.appendChild(el("div", { class: "empty-state" }, [
        icon("folder"),
        el("p", {}, "No projects yet."),
        el("button", { class: "btn primary", onclick: () => openProjectForm() }, [icon("plus"), "Start a project"]),
      ]));
      body.appendChild(pane);
      return;
    }
    pane.appendChild(el("div", { class: "pane-header" }, [
      el("span", { class: "muted small" }, state.projects.length + " project(s)"),
      el("button", { class: "btn primary small", onclick: () => openProjectForm() }, [icon("plus"), "Start a project"]),
    ]));
    const grid = el("div", { class: "card-grid" });
    state.projects.forEach((p) => {
      const c = el("div", { class: "card" }, [
        el("h3", {}, p.name),
        el("p", {}, p.summary || p.description || ""),
        el("div", { class: "card-footer" }, [
          el("div", { class: "avatar-stack" }, (p.members || []).slice(0, 3).map((m) => el("span", { class: "avatar" }, initials(m.name)))),
          el("span", { class: "muted small" }, (p.members || []).length + " member(s)"),
        ]),
      ]);
      c.addEventListener("click", () => { currentProjectId = p.id; renderCenter(true); });
      grid.appendChild(c);
    });
    pane.appendChild(grid);
    body.appendChild(pane);
  }

  function renderProjectDetail(pane, body) {
    const project = byId(state.projects, currentProjectId);
    if (!project) { currentProjectId = null; return renderProjectsView(body); }
    pane.appendChild(el("a", { class: "back-link", onclick: () => { currentProjectId = null; renderCenter(true); } }, [icon("chevronright", "rotate180"), "All projects"]));
    const header = el("div", { class: "detail-card" });
    header.appendChild(el("h2", {}, project.name));
    if (project.summary) header.appendChild(el("p", { class: "detail-tagline" }, project.summary));
    header.appendChild(el("p", { class: "detail-desc" }, project.description || ""));
    header.appendChild(el("div", { class: "action-row" }, [
      el("button", { class: "btn primary small", onclick: () => createChat(project.id) }, [icon("chat"), "Start a chat"]),
      el("button", { class: "btn small", onclick: () => openInviteForm(project) }, [icon("plus"), "Invite collaborator"]),
    ]));
    pane.appendChild(header);

    pane.appendChild(el("div", { class: "section-title" }, "Members"));
    pane.appendChild(el("div", { class: "list-rows" }, (project.members || []).map((m) => el("div", { class: "list-row" }, [
      el("div", { class: "row-main row-title" }, [el("span", { class: "avatar" }, initials(m.name)), m.name]),
      el("span", { class: "badge pill" }, m.status || m.role),
    ]))));

    const memory = state.memory.filter((m) => m.projectId === project.id);
    pane.appendChild(el("div", { class: "section-title" }, "Project memory"));
    pane.appendChild(memory.length
      ? el("div", { class: "list-rows" }, memory.map((m) => el("div", { class: "list-row" }, [el("div", { class: "row-main" }, [el("div", { class: "row-title" }, m.title), el("div", { class: "row-sub" }, m.content)])])))
      : el("p", { class: "muted small" }, "No project memory yet."));

    const meetings = state.meetings.filter((m) => m.projectId === project.id);
    pane.appendChild(el("div", { class: "section-title" }, "Meetings"));
    pane.appendChild(meetings.length
      ? el("div", { class: "list-rows" }, meetings.map((m) => { const row = el("div", { class: "list-row", style: "cursor:pointer" }, [el("div", { class: "row-main" }, [el("div", { class: "row-title" }, m.title), el("div", { class: "row-sub" }, m.summary)])]); row.addEventListener("click", () => { view = "meetings"; currentMeetingId = m.id; $$(".nav-item[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === "meetings")); renderCenter(true); }); return row; }))
      : el("p", { class: "muted small" }, "No meetings yet."));

    const files = state.artifacts.filter((a) => a.projectId === project.id);
    pane.appendChild(el("div", { class: "section-title" }, "Files"));
    pane.appendChild(files.length
      ? el("div", { class: "list-rows" }, files.map((a) => el("div", { class: "list-row" }, [
          el("div", { class: "row-main" }, [el("div", { class: "row-title" }, a.name), el("div", { class: "row-sub" }, fmtSize(a.size))]),
          el("a", { class: "icon-btn", href: `${API}/artifacts/${a.id}/download` }, icon("download")),
        ])))
      : el("p", { class: "muted small" }, "No files yet."));
    body.appendChild(pane);
  }

  function openProjectForm() {
    const nameInput = el("input", { required: "required" });
    const descInput = el("textarea", {});
    const body = el("div", { class: "simple-form" }, [
      el("div", { class: "field" }, [el("label", {}, "Name"), nameInput]),
      el("div", { class: "field" }, [el("label", {}, "Description"), descInput]),
    ]);
    const createBtn = el("button", { class: "btn primary" }, "Create project");
    const { close } = openModal({ title: "Start a project", body, footer: [el("button", { class: "btn ghost" }, "Cancel"), createBtn] });
    $$("#modal-root .modal-footer .btn.ghost").forEach((b) => b.addEventListener("click", close));
    createBtn.addEventListener("click", async () => {
      if (!nameInput.value.trim()) return;
      try {
        await api("/records/projects", { method: "POST", body: { data: { name: nameInput.value.trim(), description: descInput.value.trim(), members: [{ name: state.user.name, role: "Owner" }], scope: "Project" } } });
        close(); await refresh();
      } catch (err) { alert(err.message); }
    });
  }

  function openInviteForm(project) {
    const nameInput = el("input", { placeholder: "Full name", required: "required" });
    const roleSelect = el("select", {}, ["Scientist", "Admin", "Viewer"].map((r) => el("option", {}, r)));
    const body = el("div", { class: "simple-form" }, [
      el("div", { class: "field" }, [el("label", {}, "Name"), nameInput]),
      el("div", { class: "field" }, [el("label", {}, "Role"), roleSelect]),
      el("p", { class: "form-note" }, "This demo has no company directory yet — invites are recorded on the project but no email is sent."),
    ]);
    const sendBtn = el("button", { class: "btn primary" }, "Send invite");
    const { close } = openModal({ title: "Invite a collaborator", body, footer: [el("button", { class: "btn ghost" }, "Cancel"), sendBtn] });
    $$("#modal-root .modal-footer .btn.ghost").forEach((b) => b.addEventListener("click", close));
    sendBtn.addEventListener("click", async () => {
      if (!nameInput.value.trim()) return;
      const members = [...(project.members || []), { name: nameInput.value.trim(), role: roleSelect.value, status: "Invited — pending" }];
      try {
        await api(`/records/projects/${project.id}`, { method: "PATCH", body: { data: { members } } });
        close(); await refresh();
      } catch (err) { alert(err.message); }
    });
  }

  // ==================== MEMORY ====================
  function renderMemoryView(body) {
    body.innerHTML = "";
    const pane = el("div", { class: "pane" });
    pane.appendChild(el("p", { class: "pane-intro" }, "Personal, project, team and company context the assistant is allowed to use."));

    pane.appendChild(el("div", { class: "pane-header" }, [
      el("span", {}),
      el("button", { class: "btn primary small", onclick: () => openMemoryForm() }, [icon("plus"), "New memory"]),
    ]));

    const fileInput = el("input", { type: "file", multiple: "multiple", hidden: "hidden" });
    fileInput.addEventListener("change", () => Array.from(fileInput.files).forEach((f) => attachFile(null, f, null)));
    const dropzone = el("div", { class: "dropzone", onclick: () => fileInput.click() }, [
      icon("upload"), el("div", {}, [el("b", {}, "Drop files here"), " or click to browse"]),
      el("div", { class: "small" }, "Text, CSV, FASTA, PDF, and Office documents"),
      fileInput,
    ]);
    ["dragover", "dragenter"].forEach((evt) => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((evt) => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
    dropzone.addEventListener("drop", (e) => Array.from(e.dataTransfer.files).forEach((f) => attachFile(null, f, null)));
    pane.appendChild(dropzone);

    const files = state.artifacts.filter((a) => !a.chatId);
    pane.appendChild(el("div", { class: "section-title" }, "Files"));
    pane.appendChild(files.length
      ? el("div", { class: "list-rows" }, files.map((a) => el("div", { class: "list-row" }, [
          el("div", { class: "row-main" }, [el("div", { class: "row-title" }, a.name + (a.scanStatus === "review" ? " ⚠" : "")), el("div", { class: "row-sub" }, fmtSize(a.size) + (a.projectId ? " · " + (byId(state.projects, a.projectId)?.name || "Project") : " · Personal"))]),
          el("a", { class: "icon-btn", href: `${API}/artifacts/${a.id}/download` }, icon("download")),
        ])))
      : el("p", { class: "muted small" }, "No files yet."));

    pane.appendChild(el("div", { class: "section-title" }, "Current memory"));
    const scopes = ["Personal", "Project", "Team", "Company"];
    const anyMemory = state.memory.length > 0;
    if (!anyMemory) pane.appendChild(el("p", { class: "muted small" }, "No memory saved yet."));
    scopes.forEach((scope) => {
      const items = state.memory.filter((m) => (m.scope || "Personal") === scope);
      if (!items.length) return;
      pane.appendChild(el("div", { class: "row-sub", style: "margin:14px 0 6px;font-weight:600;color:var(--text-secondary)" }, scope));
      pane.appendChild(el("div", { class: "list-rows" }, items.map((m) => {
        const row = el("div", { class: "list-row" }, [
          el("div", { class: "row-main" }, [el("div", { class: "row-title" }, m.title), el("div", { class: "row-sub" }, m.content)]),
          el("div", { class: "row-actions" }, [el("button", { class: "icon-btn", title: "Delete", onclick: async () => { await api(`/records/memory/${m.id}`, { method: "DELETE" }); await refresh(); } }, icon("trash"))]),
        ]);
        return row;
      })));
    });
    body.appendChild(pane);
  }

  function openMemoryForm() {
    const titleInput = el("input", { required: "required" });
    const contentInput = el("textarea", { required: "required" });
    const scopeSelect = el("select", {}, ["Personal", "Project", "Team", "Company"].map((s) => el("option", {}, s)));
    const projSelect = el("select", {}, state.projects.map((p) => el("option", { value: p.id }, p.name)));
    const projRow = el("div", { class: "field" }, [el("label", {}, "Project"), projSelect]);
    scopeSelect.addEventListener("change", () => { projRow.hidden = scopeSelect.value !== "Project"; });
    projRow.hidden = true;
    const body = el("div", { class: "simple-form" }, [
      el("div", { class: "field" }, [el("label", {}, "Title"), titleInput]),
      el("div", { class: "field" }, [el("label", {}, "Content"), contentInput]),
      el("div", { class: "field" }, [el("label", {}, "Scope"), scopeSelect]),
      projRow,
    ]);
    const createBtn = el("button", { class: "btn primary" }, "Save");
    const { close } = openModal({ title: "New memory", body, footer: [el("button", { class: "btn ghost" }, "Cancel"), createBtn] });
    $$("#modal-root .modal-footer .btn.ghost").forEach((b) => b.addEventListener("click", close));
    createBtn.addEventListener("click", async () => {
      if (!titleInput.value.trim() || !contentInput.value.trim()) return;
      const scope = scopeSelect.value;
      try {
        await api("/records/memory", { method: "POST", body: { data: { title: titleInput.value.trim(), content: contentInput.value.trim(), scope, projectId: scope === "Project" ? projSelect.value : null } } });
        close(); await refresh();
      } catch (err) { alert(err.message); }
    });
  }

  // ==================== MEETINGS ====================
  function renderMeetingsView(body, force) {
    // The poll above re-renders every 1.8 seconds. Without this check the
    // whole view is torn down and rebuilt each time, which throws away the
    // transcript's scroll position — scroll into a long transcript and it
    // snaps back to the top before you can read a line. Rebuild only when
    // something actually changed, the way the chat view already does.
    const sig = "meetings:" + currentMeetingId + "|" + (recording ? "rec" : "") + "|"
      + recordingJobs.map((j) => j.id + j.status + j.stage).join(",") + "|"
      + state.meetings.map((m) => m.id + (m.title || "") + (m.summary || "").length).join(",");
    if (sig === lastRenderSig.center && !force) return;
    lastRenderSig.center = sig;

    body.innerHTML = "";
    const pane = el("div", { class: "pane" });
    if (currentMeetingId) {
      const meeting = byId(state.meetings, currentMeetingId);
      if (!meeting) { currentMeetingId = null; return renderMeetingsView(body, true); }
      pane.appendChild(el("a", { class: "back-link", onclick: () => { currentMeetingId = null; renderCenter(true); } }, [icon("chevronright", "rotate180"), "All meetings"]));
      const card = el("div", { class: "detail-card" });
      card.appendChild(el("h2", {}, meeting.title));
      const facts = [];
      if (meeting.recordedAt) facts.push(new Date(meeting.recordedAt).toLocaleString());
      if (meeting.durationMs) facts.push(Recorder.formatClock(meeting.durationMs / 1000));
      if (meeting.wordCount) facts.push(meeting.wordCount.toLocaleString() + " words");
      if (facts.length) card.appendChild(el("div", { class: "row-sub" }, facts.join(" · ")));
      (meeting.caveats || []).forEach((note) =>
        card.appendChild(el("div", { class: "meeting-caveat" }, note)));
      const speakerPanel = renderSpeakerNames(meeting);
      if (speakerPanel) card.appendChild(speakerPanel);
      card.appendChild(el("p", { class: "detail-desc" }, meeting.summary || ""));
      // Only sections that have something in them. A recording never has
      // speaker names or extracted deadlines, so printing "None extracted."
      // three times buries the sections that do say something — and the
      // caveat above already explains why the people list is empty.
      const block = (label, items) => el("div", { style: "margin-top:14px" }, [
        el("div", { class: "section-title", style: "margin:0 0 6px" }, label),
        el("ul", { style: "margin:0;padding-left:18px" }, items.map((i) => el("li", {}, i))),
      ]);
      const sections = [
        ["Decisions", meeting.decisions], ["Action items", meeting.actions],
        ["Deadlines", meeting.deadlines], ["Open questions", meeting.questions],
        ["Findings", meeting.findings], ["People", meeting.people],
      ].filter(([, items]) => (items || []).length);
      if (sections.length) sections.forEach(([label, items]) => card.appendChild(block(label, items)));
      else card.appendChild(el("p", { class: "muted small", style: "margin-top:14px" },
        "Nothing was extracted from this meeting. The transcript below is still complete."));
      if (meeting.transcript) {
        card.appendChild(el("div", { class: "section-title" }, "Transcript"));
        card.appendChild(el("div", { class: "transcript-box" }, meeting.transcript));
      }
      // How this meeting was produced, in the meeting's own words rather than
      // one fixed sentence: a recording transcribed locally and a transcript
      // somebody pasted in have very different reliability, and the reader
      // cannot tell them apart otherwise.
      card.appendChild(el("p", { class: "muted small", style: "margin-top:12px" },
        meeting.method || "Rule-based extraction — review before treating as verified minutes."));
      pane.appendChild(card);
      body.appendChild(pane);
      return;
    }
    pane.appendChild(el("p", { class: "pane-intro" }, "Record a meeting on this device, or import a transcript. Decisions and action items are pulled out automatically, and the assistant can answer questions about any meeting here."));
    pane.appendChild(renderRecordBar());
    if (recording) pane.appendChild(renderLivePreview());
    pane.appendChild(renderVoicesBar());
    recordingJobs.forEach((job) => pane.appendChild(renderJobRow(job)));

    if (!state.meetings.length) {
      pane.appendChild(el("div", { class: "empty-state" }, [icon("calendar"), el("p", {}, "No meetings yet. Press Record to capture one, or attach a transcript file from a chat.")]));
      body.appendChild(pane);
      return;
    }
    const grid = el("div", { class: "card-grid" });
    [...state.meetings]
      .sort((a, b) => (b.recordedAt || "").localeCompare(a.recordedAt || ""))
      .forEach((m) => {
        const meta = [];
        if (m.recordedAt) meta.push(new Date(m.recordedAt).toLocaleString());
        if (m.durationMs) meta.push(Recorder.formatClock(m.durationMs / 1000));
        const c = el("div", { class: "card" }, [
          el("h3", {}, m.title),
          meta.length ? el("div", { class: "row-sub" }, meta.join(" · ")) : null,
          el("p", {}, m.summary || ""),
        ]);
        c.addEventListener("click", () => { currentMeetingId = m.id; renderCenter(true); });
        grid.appendChild(c);
      });
    pane.appendChild(grid);
    body.appendChild(pane);
  }

  // Diarisation can tell voices apart but has no idea whose they are, so it
  // hands back "Speaker 1" and somebody has to say who that is. Without this
  // the labels are nearly useless: an action item owned by "Speaker 2" is no
  // more use than one owned by nobody.
  function renderSpeakerNames(meeting) {
    const labels = [...new Set((meeting.transcript || "").match(/^Speaker \d+(?=:)/gm) || [])]
      .sort((a, b) => parseInt(a.slice(8), 10) - parseInt(b.slice(8), 10));
    if (!labels.length) return null;

    const panel = el("div", { class: "speaker-panel" });
    panel.appendChild(el("div", { class: "section-title", style: "margin:0 0 8px" }, "Who was speaking"));
    panel.appendChild(el("p", { class: "muted small", style: "margin:0 0 10px" },
      "Type a name to replace the label everywhere in this meeting — transcript, decisions and action items."));
    const inputs = labels.map((label) => {
      const input = el("input", { placeholder: label, "data-label": label });
      return el("label", { class: "speaker-row" }, [el("span", { class: "muted small" }, label), input]);
    });
    panel.appendChild(el("div", { class: "speaker-grid" }, inputs));

    const status = el("span", { class: "muted small" });
    const save = el("button", { class: "btn small", onclick: async () => {
      const names = {};
      $$(".speaker-row input", panel).forEach((i) => {
        if (i.value.trim()) names[i.dataset.label] = i.value.trim();
      });
      if (!Object.keys(names).length) { status.textContent = "Type at least one name."; return; }
      save.disabled = true; status.textContent = "Saving…";
      try {
        await api(`/meetings/${meeting.id}/speakers`, { method: "POST", body: { names } });
        await refresh();
        renderCenter(true);
      } catch (err) {
        status.textContent = err.message;
        save.disabled = false;
      }
    } }, "Rename");
    panel.appendChild(el("div", { class: "speaker-actions" }, [save, status]));
    return panel;
  }

  // ==================== RECORDING ====================
  const Recorder = window.BedrockRecorder;
  let recordingUI = null;     // live DOM nodes the timer/meter write into
  let readiness = null;       // cached /meetings/readiness answer

  async function loadReadiness() {
    if (readiness) return readiness;
    try { readiness = await api("/meetings/readiness"); }
    catch { readiness = { ready: false, issues: ["The server did not answer the readiness check."] }; }
    return readiness;
  }

  function renderRecordBar() {
    const bar = el("div", { class: "record-bar" });

    if (!Recorder || !Recorder.supported) {
      bar.classList.add("record-bar-warn");
      bar.appendChild(el("p", { class: "muted small", style: "margin:0" },
        "This browser cannot record audio. Use Chrome or Edge."));
      return bar;
    }

    if (!recording) {
      const btn = el("button", { class: "btn primary", onclick: () => openRecorderSetup() }, [
        el("span", { class: "record-dot-idle" }), "Record meeting",
      ]);
      bar.appendChild(btn);
      bar.appendChild(el("p", { class: "muted small", style: "margin:0" },
        "Audio is transcribed on this server. Nothing is sent to any outside service."));
      return bar;
    }

    // Live. These nodes are written to directly by the session callbacks
    // below, sixty times a second, so they must not be rebuilt by a re-render.
    bar.classList.add("record-bar-live");
    const clock = el("span", { class: "record-clock" }, "00:00");
    const meterFill = el("div", { class: "record-meter-fill" });
    const silence = el("span", { class: "record-silence", hidden: true }, "No sound detected");
    const stopBtn = el("button", { class: "btn danger", onclick: () => finishRecording() }, "Stop and save");
    bar.appendChild(el("span", { class: "record-dot-live" }));
    bar.appendChild(el("div", { class: "record-live-info" }, [
      el("div", { class: "record-live-title" }, recording.meetingTitle),
      el("div", { class: "record-live-sub" }, [clock, el("span", {}, " · "), el("span", {}, recording.sourceLabel), silence]),
    ]));
    bar.appendChild(el("div", { class: "record-meter" }, meterFill));
    bar.appendChild(stopBtn);
    recordingUI = { clock, meterFill, silence, stopBtn, preview: null };
    return bar;
  }

  // Diarisation tells voices apart but has no idea whose they are. One ten-
  // second sample each, once, and meetings name people by themselves.
  let voices = null;
  async function loadVoices() {
    try { voices = await api("/meetings/voices"); } catch { voices = { available: false, voices: [] }; }
    return voices;
  }

  function renderVoicesBar() {
    const bar = el("div", { class: "voices-bar" });
    if (!voices) { loadVoices().then(() => { if (view === "meetings") renderCenter(true); }); return bar; }
    if (!voices.available) return bar;

    const names = (voices.voices || []).map((v) =>
      el("span", { class: "voice-chip" }, [
        el("span", {}, v.name),
        el("button", { class: "link-btn small", title: "Forget this voice", onclick: async () => {
          await api(`/meetings/voices/${v.id}`, { method: "DELETE" });
          await loadVoices(); renderCenter(true);
        } }, "×"),
      ]));

    bar.appendChild(el("div", { class: "voices-label" }, [
      el("span", { class: "section-title", style: "margin:0" }, "Recognised voices"),
      el("span", { class: "muted small" },
        names.length ? "Meetings label these people automatically."
                     : "Nobody yet — meetings will say “Speaker 1” until somebody is saved."),
    ]));
    if (names.length) bar.appendChild(el("div", { class: "voices-list" }, names));
    bar.appendChild(el("button", { class: "btn ghost small", onclick: () => openVoiceEnroll() }, "Add a voice"));
    return bar;
  }

  async function openVoiceEnroll() {
    const body = el("div", { class: "simple-form" });
    const nameInput = el("input", { placeholder: "e.g. Priya Raman" });
    const status = el("p", { class: "muted small" }, "Read the line below for about ten seconds.");
    const line = el("p", { class: "enroll-line" },
      "“This is my voice. I am saving it so meetings can label what I say, instead of calling me Speaker Two.”");
    const meterFill = el("div", { class: "record-meter-fill" });
    const error = el("p", { class: "form-error", hidden: true });
    body.appendChild(field("Name", nameInput));
    body.appendChild(line);
    body.appendChild(el("div", { class: "record-meter" }, meterFill));
    body.appendChild(status);
    body.appendChild(el("p", { class: "muted small" },
      "The recording is not kept — only a numeric voice print, which cannot be played back. It is not a password: it identifies who is speaking, never what anyone may read."));
    body.appendChild(error);

    let session = null;
    const startBtn = el("button", { class: "btn primary" }, "Start");
    const dialog = openModal({ title: "Save a voice", body, footer: startBtn });

    startBtn.addEventListener("click", async () => {
      error.hidden = true;
      if (!nameInput.value.trim()) { error.textContent = "Give the person a name."; error.hidden = false; return; }
      if (!session) {
        try {
          session = new Recorder.Session({
            useMic: true, useSystem: false, roomMode: false, deviceId: null,
            onLevel: (l) => { meterFill.style.width = Math.round(l * 100) + "%"; },
            onTick: (s) => { status.textContent = `Recording — ${Recorder.formatClock(s)}. Ten seconds is plenty.`; },
            onEnded: () => {}, onError: (m) => { error.textContent = m; error.hidden = false; },
          });
          await session.start();
          startBtn.textContent = "Save";
        } catch (err) { error.textContent = err.message; error.hidden = false; session = null; }
        return;
      }
      startBtn.disabled = true; startBtn.textContent = "Saving…";
      const result = await session.stop();
      try {
        const form = new FormData();
        form.append("audio", result.blob, "sample.webm");
        form.append("name", nameInput.value.trim());
        const res = await fetch("/workspace/api/meetings/voices", { method: "POST", body: form, credentials: "same-origin" });
        if (!res.ok) throw new Error((await res.json()).detail || "That sample could not be saved.");
        await loadVoices();
        dialog.close();
        renderCenter(true);
      } catch (err) {
        error.textContent = err.message; error.hidden = false;
        startBtn.disabled = false; startBtn.textContent = "Start"; session = null;
      }
    });
  }

  // Words while the meeting is still happening. Explicitly provisional: no
  // speaker names, nothing corrected, and replaced entirely by the real
  // transcript when the recording is saved.
  function renderLivePreview() {
    const box = el("div", { class: "live-preview" });
    box.appendChild(el("div", { class: "live-preview-head" }, [
      el("span", { class: "section-title", style: "margin:0" }, "Live preview"),
      el("span", { class: "muted small" },
        "Rough, and without speaker names. The full transcript is made when you stop."),
    ]));
    const text = el("div", { class: "live-preview-text muted" }, livePreviewText || "Listening…");
    box.appendChild(text);
    if (recordingUI) recordingUI.preview = text;
    return box;
  }

  function renderJobRow(job) {
    const row = el("div", { class: "record-job" });
    const done = job.status === "done";
    const failed = job.status === "failed";
    row.appendChild(el("div", { class: "record-job-main" }, [
      el("div", { class: "row-title" }, job.title),
      el("div", { class: "row-sub" },
        failed ? (job.error || "Failed")
          : done ? "Saved" + (job.notes && job.notes.length ? " — " + job.notes.join(" ") : "")
          : job.stage + "…"),
    ]));
    if (!done && !failed) row.appendChild(el("div", { class: "record-job-spinner" }));
    if (done) {
      row.appendChild(el("button", { class: "btn ghost small", onclick: async () => {
        await refresh();
        currentMeetingId = job.meetingId;
        renderCenter(true);
      } }, "Open"));
    }
    if (done || failed) {
      row.appendChild(el("button", { class: "icon-btn", title: "Dismiss", onclick: () => {
        recordingJobs = recordingJobs.filter((j) => j.id !== job.id);
        renderCenter(true);
      } }, icon("close")));
    }
    return row;
  }

  async function openRecorderSetup() {
    const ready = await loadReadiness();
    const body = el("div", { class: "simple-form" });

    if (!ready.ready) {
      body.appendChild(el("p", { class: "muted" },
        "This server cannot transcribe recordings yet. Ask whoever runs it to fix the following, then reload:"));
      body.appendChild(el("ul", { class: "muted small" }, ready.issues.map((i) => el("li", {}, i))));
      openModal({ title: "Recording is not set up", body });
      return;
    }

    const titleInput = el("input", { placeholder: "e.g. KRAS research weekly" });
    const micSelect = el("select", {});
    const useMic = el("input", { type: "checkbox", checked: true });
    const useSystem = el("input", { type: "checkbox" });
    const notesInput = el("textarea", { rows: "3", placeholder: "Optional. Anything you jot here is rewritten into the notes using the transcript as the source of fact." });

    const projSelect = el("select", {}, [el("option", { value: "" }, "Personal — no project")]);
    (state.projects || []).forEach((p) => projSelect.appendChild(el("option", { value: p.id }, p.name)));

    micSelect.appendChild(el("option", { value: "" }, "Default microphone"));
    try {
      // Labels are blank until the page has been granted microphone access
      // once; the default entry above still works, so this is not fatal.
      (await Recorder.listMicrophones()).forEach((d, i) => {
        if (d.deviceId && d.deviceId !== "default") {
          micSelect.appendChild(el("option", { value: d.deviceId }, d.label || `Microphone ${i + 1}`));
        }
      });
    } catch (_) {}

    const field = (label, control, hint) => el("div", { class: "field" }, [
      el("label", {}, label), control,
      hint ? el("div", { class: "muted small" }, hint) : null,
    ]);
    const check = (control, label, hint) => el("label", { class: "check-row" }, [
      control, el("div", {}, [el("div", {}, label), hint ? el("div", { class: "muted small" }, hint) : null]),
    ]);

    const speakerInput = el("input", { type: "number", min: "1", max: "20", placeholder: "e.g. 4" });
    body.appendChild(field("Meeting name", titleInput));
    body.appendChild(field("Project", projSelect));
    body.appendChild(field("How many people will be speaking?", speakerInput,
      "Worth filling in. Left to guess from the audio alone it over-counts badly when people talk over each other — on a real three-person call it found twenty-four. Given the number, it is reliable. You can rename the speakers afterwards."));
    const roomMode = el("input", { type: "checkbox", checked: true });
    body.appendChild(check(useMic, "This device's microphone", "For a meeting in a room, and for your own voice on a call."));
    body.appendChild(check(roomMode, "Several people in the room",
      "Turns off the noise suppression and echo cancellation a laptop applies by default. Both are tuned for one person close to the microphone and treat a voice across the room as noise to remove — which is the voice you want. Turn this off if you are recording only yourself."));
    body.appendChild(check(useSystem, "Audio from an online meeting",
      "Captures a browser tab. Chrome will ask which to share — pick the meeting tab and tick “Also share tab audio”. On macOS whole-screen audio is not available, so join the call in a browser tab, or select a loopback device above."));
    body.appendChild(field("Microphone", micSelect));
    body.appendChild(field("Your notes during the meeting", notesInput));
    const consent = el("input", { type: "checkbox" });
    const whoElse = el("input", { placeholder: "e.g. Luke, Priya, Ravi" });
    body.appendChild(check(consent, "Everyone present knows they are being recorded",
      "In several US states every party has to agree. This records that you said so, with your name and the time — it is your statement, not proof."));
    body.appendChild(field("Who else is here", whoElse,
      "Optional. Stored with the meeting so there is a record of who was in the room."));

    const startBtn = el("button", { class: "btn primary" }, "Start recording");
    const error = el("p", { class: "form-error", hidden: true });
    body.appendChild(error);

    const dialog = openModal({ title: "Record a meeting", body, footer: startBtn });
    startBtn.addEventListener("click", async () => {
      error.hidden = true;
      startBtn.disabled = true;
      startBtn.textContent = "Starting…";
      try {
        await beginRecording({
          title: titleInput.value.trim(),
          projectId: projSelect.value || null,
          notes: notesInput.value.trim() || null,
          useMic: useMic.checked,
          useSystem: useSystem.checked,
          roomMode: roomMode.checked,
          deviceId: micSelect.value || null,
          expectedSpeakers: parseInt(speakerInput.value, 10) || null,
          consentConfirmed: consent.checked,
          participants: whoElse.value.trim() || null,
        });
        dialog.close();
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
        startBtn.disabled = false;
        startBtn.textContent = "Start recording";
      }
    });
  }

  async function beginRecording(opts) {
    if (!opts.useMic && !opts.useSystem) throw new Error("Choose at least one audio source.");

    let quietSince = null;
    const session = new Recorder.Session({
      useMic: opts.useMic,
      useSystem: opts.useSystem,
      roomMode: opts.roomMode,
      deviceId: opts.deviceId,
      onLevel: (level) => {
        if (!recordingUI) return;
        recordingUI.meterFill.style.width = Math.round(level * 100) + "%";
        // Ten seconds of near-silence is worth saying out loud. A muted
        // microphone is the one failure that cannot be recovered afterwards.
        if (level > 0.03) { quietSince = null; recordingUI.silence.hidden = true; }
        else {
          if (quietSince === null) quietSince = Date.now();
          if (Date.now() - quietSince > 10000) recordingUI.silence.hidden = false;
        }
      },
      onTick: (seconds) => { if (recordingUI) recordingUI.clock.textContent = Recorder.formatClock(seconds); },
      onEnded: (reason) => { finishRecording(reason); },
      onError: (message) => { finishRecording(message); },
    });

    await session.start();

    // Preview is best-effort: a meeting must never fail to record because the
    // words could not be shown while it happened.
    livePreviewText = "";
    liveSessionId = null;
    try {
      liveSessionId = (await api("/meetings/live", { method: "POST", body: {} })).id;
      session.startPreview(async (clip) => {
        if (!liveSessionId) return;
        try {
          const state = await Recorder.sendClip(liveSessionId, clip);
          livePreviewText = state.text;
          if (recordingUI && recordingUI.preview) {
            recordingUI.preview.textContent = livePreviewText || "Listening…";
          }
        } catch (_) { /* a dropped clip costs nothing; the recording is intact */ }
      });
    } catch (_) { liveSessionId = null; }

    session.meetingTitle = opts.title || "Untitled meeting";
    session.sourceLabel = opts.useMic && opts.useSystem ? "microphone + meeting audio"
      : opts.useSystem ? "meeting audio" : "microphone";
    session.options = opts;
    recording = session;
    window.addEventListener("beforeunload", warnWhileRecording);
    renderCenter(true);
  }

  function warnWhileRecording(e) {
    if (!recording) return;
    e.preventDefault();
    e.returnValue = "";
  }

  async function finishRecording(reason) {
    const session = recording;
    if (!session) return;
    recording = null;
    recordingUI = null;
    window.removeEventListener("beforeunload", warnWhileRecording);

    const result = await session.stop();
    if (liveSessionId) {
      const finished = liveSessionId;
      liveSessionId = null;
      api(`/meetings/live/${finished}`, { method: "DELETE" }).catch(() => {});
    }
    livePreviewText = "";
    renderCenter(true);
    if (!result || !result.blob.size) {
      alert(reason || "Nothing was recorded.");
      return;
    }

    const durationSeconds = (Date.now() - result.startedAt.getTime()) / 1000;
    const pending = {
      id: "upload-" + Date.now(),
      title: session.meetingTitle,
      status: "running",
      stage: "Uploading",
      notes: [],
    };
    recordingJobs = [pending, ...recordingJobs];
    renderCenter(true);

    try {
      const job = await Recorder.upload({
        blob: result.blob,
        startedAt: result.startedAt,
        mime: result.mime,
        title: session.meetingTitle,
        projectId: session.options.projectId,
        notes: session.options.notes,
        expectedSpeakers: session.options.expectedSpeakers,
        consentConfirmed: session.options.consentConfirmed,
        participants: session.options.participants,
        durationSeconds,
        onProgress: (fraction) => {
          pending.stage = "Uploading " + Math.round(fraction * 100) + "%";
          const row = recordingJobs.find((j) => j.id === pending.id);
          if (row) row.stage = pending.stage;
          renderCenter(true);
        },
      });
      recordingJobs = recordingJobs.map((j) => (j.id === pending.id ? job : j));
      startJobPolling();
    } catch (err) {
      recordingJobs = recordingJobs.map((j) =>
        j.id === pending.id ? { ...j, status: "failed", stage: "Failed", error: err.message } : j);
    }
    renderCenter(true);
  }

  function startJobPolling() {
    if (recordingJobPoll) return;
    recordingJobPoll = setInterval(async () => {
      const live = recordingJobs.filter((j) => j.status !== "done" && j.status !== "failed" && !String(j.id).startsWith("upload-"));
      if (!live.length) { clearInterval(recordingJobPoll); recordingJobPoll = null; return; }
      let changed = false;
      for (const job of live) {
        try {
          const updated = await api(`/meetings/recordings/${job.id}`);
          if (updated.status !== job.status || updated.stage !== job.stage) changed = true;
          recordingJobs = recordingJobs.map((j) => (j.id === job.id ? updated : j));
          // A finished transcription added a meeting; pull it into `state` so
          // the card appears without the person having to reload.
          if (updated.status === "done") await refresh();
        } catch (_) { /* transient; the next tick retries */ }
      }
      if (changed && view === "meetings") renderCenter(true);
    }, 2000);
  }

  // ==================== SETTINGS ====================
  function renderSettingsView(body) {
    body.innerHTML = "";
    const pane = el("div", { class: "pane" });
    const cfg = state.config || {};
    const prefs = (state.preferences || [])[0] || {};

    const row = (title, sub, valueNode) => el("div", { class: "settings-row" }, [
      el("div", {}, [el("div", { class: "row-title" }, title), sub ? el("div", { class: "row-sub" }, sub) : null]),
      valueNode,
    ]);

    pane.appendChild(el("div", { class: "settings-group" }, [
      el("h2", {}, "Account"),
      row("Name", null, el("span", { class: "row-value" }, state.user.name)),
      row("Sign out", "End this session on this device", el("button", { class: "btn small", onclick: () => $("#logout-btn").click() }, "Sign out")),
    ]));

    pane.appendChild(el("div", { class: "settings-group" }, [
      el("h2", {}, "Environment"),
      row("Mode", null, el("span", { class: "badge pill" }, cfg.demo ? "Demo (fictional data)" : "Real")),
      row("AI provider", null, el("span", { class: "row-value" }, cfg.ai || "Unknown")),
      row("Authentication", null, el("span", { class: "row-value" }, cfg.auth || "Unknown")),
      row("Appearance", null, el("span", { class: "row-value" }, "Dark")),
    ]));

    const buddySwitch = el("label", { class: "switch" }, [el("input", { type: "checkbox", checked: prefs.buddy ? "checked" : null }), el("span", { class: "track" })]);
    buddySwitch.querySelector("input").addEventListener("change", async (e) => {
      await api("/records/preferences", { method: "POST", body: { data: { ...prefs, buddy: e.target.checked } } });
      await refresh();
    });
    pane.appendChild(el("div", { class: "settings-group" }, [
      el("h2", {}, "Preferences"),
      row("Buddy assistant", "A lightweight helper for navigating the workspace", buddySwitch),
      row("Data retention", null, el("span", { class: "row-value" }, prefs.retention || "Not set")),
      row("Scientific vocabulary", "Terms the assistant treats as known jargon", el("span", { class: "row-value" }, (prefs.vocabulary || "").split("\n").filter(Boolean).join(", "))),
    ]));

    pane.appendChild(el("div", { class: "settings-group" }, [
      el("h2", {}, "Company Knowledge connectors"),
      ...(cfg.connectors || []).map((c) => row(c.name, null, el("span", { class: "badge pill" }, c.status))),
    ]));
    body.appendChild(pane);
  }

  // ==================== ACTIVITY (agents only) ====================
  const expandedFollowups = new Set();

  function renderActivity() {
    const runs = [...state.runs].filter((r) => r.mode && !["ask", "auto"].includes(r.mode)).sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
    const sig = JSON.stringify(runs.map((r) => [r.id, r.status, r.currentStep, (r.steps || []).map((s) => s.status), (r.followups || []).map((f) => [f.id, f.status]), expandedFollowups.has(r.id)]));
    if (sig === lastRenderSig.activity) return;
    lastRenderSig.activity = sig;
    const host = $("#activity-body");
    host.innerHTML = "";
    if (!runs.length) { host.appendChild(el("div", { class: "empty-state" }, [icon("agent"), el("p", {}, "No agent activity yet. Use Create Agent from a chat to start one.")])); return; }
    runs.forEach((r) => {
      const chat = byId(state.chats, r.chatId);
      const card = el("div", { class: "run-card" });
      card.appendChild(el("div", { class: "run-title" }, r.title || r.prompt));
      card.appendChild(el("div", { class: "run-meta" }, [el("span", { class: "badge status-" + r.status }, r.status), el("span", {}, r.role || "")]));
      if (r.steps && r.steps.length) {
        const list = el("div", { class: "run-step-list" });
        r.steps.forEach((s) => list.appendChild(el("div", { class: "run-step " + s.status }, [el("span", { class: "dot" }), el("span", {}, s.title)])));
        card.appendChild(list);
      } else if (["queued", "running"].includes(r.status)) {
        card.appendChild(el("div", { class: "run-current-step" }, [el("span", { class: "dot" }), el("span", {}, r.currentStep || "Working…")]));
      }
      if (r.status === "waiting") {
        card.appendChild(el("div", { class: "run-actions" }, [
          el("button", { class: "btn primary small", onclick: () => decide(r.id, "approve") }, "Approve"),
          el("button", { class: "btn danger small", onclick: () => decide(r.id, "reject") }, "Reject"),
        ]));
        if (r.preview) card.appendChild(el("p", { class: "small muted" }, r.preview));
      } else if (["queued", "running"].includes(r.status)) {
        card.appendChild(el("div", { class: "run-actions" }, [el("button", { class: "btn ghost small", onclick: () => cancelRun(r.id) }, "Cancel")]));
      } else if (r.status === "failed" && r.error) {
        card.appendChild(el("p", { class: "small", style: "color:var(--bad)" }, r.error));
      }
      if (chat) card.appendChild(el("a", { class: "run-answer-link", onclick: () => {
        view = "chats"; currentChatId = chat.id;
        $$(".nav-item[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === "chats"));
        render(true);
      } }, ["View in chat", icon("chevronright")]));
      if (["completed", "failed"].includes(r.status)) card.appendChild(renderFollowupBox(r));
      host.appendChild(card);
    });
  }

  function renderFollowupBox(run) {
    const isOpen = expandedFollowups.has(run.id);
    const wrap = el("div", { class: "followup-box" });
    const toggle = el("button", { type: "button", class: "link-btn small followup-toggle", onclick: () => { isOpen ? expandedFollowups.delete(run.id) : expandedFollowups.add(run.id); lastRenderSig.activity = ""; renderActivity(); } },
      [icon(isOpen ? "chevrondown" : "chevronright"), "Ask this agent"]);
    wrap.appendChild(toggle);
    if (!isOpen) return wrap;
    const thread = el("div", { class: "followup-thread" });
    (run.followups || []).forEach((f) => {
      thread.appendChild(el("div", { class: "followup-q" }, f.prompt));
      if (f.status === "running") thread.appendChild(el("div", { class: "thinking-dots small" }, [el("span"), el("span"), el("span")]));
      else thread.appendChild(el("div", { class: "followup-a" }, f.answer || ""));
    });
    wrap.appendChild(thread);
    const pending = (run.followups || []).some((f) => f.status === "running");
    const form = el("form", { class: "followup-form" });
    const input = el("textarea", { rows: "1", placeholder: pending ? "Waiting for a reply…" : "Ask a follow-up…" });
    input.disabled = pending;
    input.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" || e.shiftKey) return;
      if (getEnterToSend()) { e.preventDefault(); form.requestSubmit(); }
    });
    const sendBtn = el("button", { type: "submit", class: "icon-btn small", disabled: pending ? "disabled" : null }, icon("send"));
    form.appendChild(input);
    form.appendChild(sendBtn);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      try {
        await api(`/runs/${run.id}/followups`, { method: "POST", body: { prompt: text } });
        lastRenderSig.activity = "";
        await refresh();
      } catch (err) { alert(err.message); }
    });
    wrap.appendChild(form);
    return wrap;
  }

  async function decide(runId, decision) {
    try { await api(`/runs/${runId}/decision`, { method: "POST", body: { decision } }); await refresh(); }
    catch (err) { alert(err.message); }
  }
  async function cancelRun(runId) {
    try { await api(`/runs/${runId}/cancel`, { method: "POST" }); await refresh(); }
    catch (err) { alert(err.message); }
  }

  const style = document.createElement("style");
  style.textContent = ".rotate180{transform:rotate(180deg)}";
  document.head.appendChild(style);

  boot();
})();
