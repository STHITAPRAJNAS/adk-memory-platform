"""
Self-contained Visual Memory & Ontology Governance Console UI (`GET /ui`).

Provides Vertex AI Memory Bank Console & Letta ADE parity out of the box with zero
external frontend build steps:
  - Tab 1: User Memory Inspector & Inline Editor (Edit / Delete / Cognitive Badges / TTL)
  - Tab 2: Letta-Style Working Memory Blocks Editor (Replace / Append)
  - Tab 3: Immutable Memory Revisions Audit Trail (ADD / UPDATE / RESOLVE / DELETE / EXPIRE / MANUAL_EDIT)
  - Tab 4: Ontology Governance & Steward Console (Domain Packs, Certify, Auto-Evolve, Audit Log)
  - Tab 5: Live Observability & Telemetry (`/v1/metrics` + `/metrics`)
"""

from __future__ import annotations

CONSOLE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agentic Composite Memory — Governance &amp; Inspector Console</title>
  <style>
    :root {
      --bg: #0b1020; --panel: #131a2e; --border: #253052;
      --text: #e6ecff; --muted: #93a3c7; --accent: #5b8cff;
      --green: #3ddc97; --amber: #ffb84d; --red: #ff6b6b; --purple: #b48eff;
    }
    * { box-sizing: border-box; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
      padding: 14px 24px; background: var(--panel); border-bottom: 1px solid var(--border); }
    header h1 { margin: 0; font-size: 17px; font-weight: 600; letter-spacing: 0.2px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    .b-blue { background: rgba(91,140,255,0.2); color: #8fb2ff; }
    .b-green { background: rgba(61,220,151,0.18); color: var(--green); }
    .b-amber { background: rgba(255,184,77,0.18); color: var(--amber); }
    .b-red { background: rgba(255,107,107,0.18); color: var(--red); }
    .b-purple { background: rgba(180,142,255,0.2); color: var(--purple); }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    input, select, textarea { background: #0c1224; border: 1px solid var(--border); color: var(--text);
      padding: 7px 10px; border-radius: 6px; font-size: 13px; }
    button { background: var(--accent); color: #fff; border: none; padding: 7px 13px; border-radius: 6px;
      font-size: 13px; font-weight: 500; cursor: pointer; }
    button:hover { opacity: 0.9; }
    button.sec { background: #233052; }
    button.danger { background: #8f2d38; }
    nav { display: flex; gap: 6px; padding: 12px 24px 0; border-bottom: 1px solid var(--border); }
    nav button { background: transparent; color: var(--muted); border-radius: 6px 6px 0 0; padding: 9px 14px; }
    nav button.active { background: var(--panel); color: var(--text); border: 1px solid var(--border); border-bottom: none; }
    main { padding: 20px 24px; max-width: 1280px; margin: 0 auto; }
    .tab { display: none; }
    .tab.active { display: block; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 14px; }
    .row { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
    .muted { color: var(--muted); font-size: 12px; }
    pre { background: #090e1c; padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 12px; border: 1px solid #1c2642; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .kpi { font-size: 26px; font-weight: 700; margin-top: 4px; color: var(--green); }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Agentic Composite Memory Console <span class="badge b-blue">v0.2.0</span></h1>
      <div class="muted">3-Tier Ontology-Grounded Memory Inspector · Revisions · Working Blocks · Governance Steward</div>
    </div>
    <div class="controls">
      <input id="appName" placeholder="tenant:app_name" value="t1:default_app" />
      <input id="userId" placeholder="user_id" value="user_1" />
      <input id="apiKey" type="password" placeholder="Optional API Key / JWT" />
      <button onclick="refreshAll()">Refresh</button>
    </div>
  </header>

  <nav>
    <button class="active" onclick="showTab('memories', this)">Memory Inspector</button>
    <button onclick="showTab('blocks', this)">Working Blocks</button>
    <button onclick="showTab('revisions', this)">Memory Revisions</button>
    <button onclick="showTab('ontology', this)">Ontology Steward</button>
    <button onclick="showTab('metrics', this)">Metrics &amp; Health</button>
  </nav>

  <main>
    <section id="tab-memories" class="tab active">
      <div class="card">
        <div class="row">
          <div>
            <strong>User Episodic &amp; Cognitive Memories</strong>
            <div class="muted">Inspect, manually edit (re-embeds + logs revision), or delete individual memories.</div>
          </div>
          <button onclick="loadMemories()">Reload Memories</button>
        </div>
      </div>
      <div id="memoriesList"></div>
    </section>

    <section id="tab-blocks" class="tab">
      <div class="card">
        <strong>Letta-Style Core Working Memory Blocks</strong>
        <div class="muted" style="margin-bottom:10px;">Self-editable blocks always injected at the top of agent recall context.</div>
        <div class="controls">
          <input id="blkLabel" placeholder="label (e.g. human_profile)" />
          <select id="blkMode"><option value="replace">replace</option><option value="append">append</option></select>
          <input id="blkValue" placeholder="block contents..." style="min-width:340px;" />
          <button onclick="saveBlock()">Save Block</button>
        </div>
      </div>
      <div id="blocksList"></div>
    </section>

    <section id="tab-revisions" class="tab">
      <div class="card row">
        <div>
          <strong>Immutable Memory Revisions Audit Trail</strong>
          <div class="muted">Every ADD, UPDATE, RESOLVE, EXPIRE, DELETE, and MANUAL_EDIT event.</div>
        </div>
        <button onclick="loadRevisions()">Reload Revisions</button>
      </div>
      <div id="revisionsList"></div>
    </section>

    <section id="tab-ontology" class="tab">
      <div class="card row">
        <div>
          <strong>Layer-1 Federated Domain Packs &amp; Overflow Evolution</strong>
          <div class="muted">Certify Domain Packs or trigger bottom-up ontology evolution from quarantined overflow.</div>
        </div>
        <div class="controls">
          <button class="sec" onclick="evolveOntology()">Evolve App Ontology</button>
          <button onclick="loadOntology()">Reload Packs &amp; Audit</button>
        </div>
      </div>
      <div id="packsList"></div>
      <div class="card"><strong>Ontology Audit Log</strong><pre id="auditLog">[]</pre></div>
    </section>

    <section id="tab-metrics" class="tab">
      <div class="grid" id="metricsCards"></div>
      <div class="card" style="margin-top:14px;"><strong>Raw Telemetry (/v1/metrics)</strong><pre id="metricsRaw">{}</pre></div>
    </section>
  </main>

  <script>
    function headers() {
      const h = { "Content-Type": "application/json" };
      const key = document.getElementById("apiKey").value.trim();
      if (key) { h["Authorization"] = "Bearer " + key; }
      return h;
    }
    function ctx() {
      return {
        app: encodeURIComponent(document.getElementById("appName").value.trim()),
        user: encodeURIComponent(document.getElementById("userId").value.trim()),
        rawApp: document.getElementById("appName").value.trim(),
        rawUser: document.getElementById("userId").value.trim(),
      };
    }
    function showTab(name, btn) {
      document.querySelectorAll(".tab").forEach(el => el.classList.remove("active"));
      document.querySelectorAll("nav button").forEach(el => el.classList.remove("active"));
      document.getElementById("tab-" + name).classList.add("active");
      btn.classList.add("active");
      refreshAll();
    }
    async function loadMemories() {
      const c = ctx();
      const res = await fetch(`/v1/memory/items?app_name=${c.app}&user_id=${c.user}`, { headers: headers() });
      const data = await res.json();
      const el = document.getElementById("memoriesList");
      const items = data.memories || [];
      if (!items.length) { el.innerHTML = '<div class="card muted">No memories found for this app/user.</div>'; return; }
      el.innerHTML = items.map(m => {
        const markers = (m.metadata?.cognitive_types || []).map(t => `<span class="badge b-purple">${t}</span>`).join(" ");
        const pinned = m.metadata?.pinned_cognitive ? '<span class="badge b-green">PINNED</span>' : '';
        const status = m.metadata?.reconciled_status ? `<span class="badge b-amber">${m.metadata.reconciled_status}</span>` : '';
        const exp = m.expires_at ? `<span class="badge b-red">TTL: ${m.expires_at.slice(0,10)}</span>` : '';
        return `<div class="card">
          <div class="row">
            <div style="flex:1;">
              <div>${pinned} ${markers} ${status} ${exp} <span class="muted">${m.author} · ${m.created_at}</span></div>
              <p style="margin:8px 0;">${escapeHtml(m.content_text)}</p>
              <div class="muted">ID: ${m.episode_id}</div>
            </div>
            <div class="controls">
              <button class="sec" onclick="editMemory('${m.episode_id}', ${JSON.stringify(m.content_text)})">Edit</button>
              <button class="danger" onclick="deleteMemory('${m.episode_id}')">Delete</button>
            </div>
          </div>
        </div>`;
      }).join("");
    }
    async function editMemory(id, prev) {
      const updated = prompt("Edit memory content:", prev);
      if (!updated || updated === prev) return;
      const c = ctx();
      await fetch(`/v1/memory/items/${id}`, {
        method: "PUT", headers: headers(),
        body: JSON.stringify({ app_name: c.rawApp, user_id: c.rawUser, new_content: updated, actor: "console_ui" })
      });
      loadMemories(); loadRevisions();
    }
    async function deleteMemory(id) {
      if (!confirm("Delete this memory record?")) return;
      const c = ctx();
      await fetch(`/v1/memory/items/${id}?app_name=${c.app}&user_id=${c.user}`, { method: "DELETE", headers: headers() });
      loadMemories(); loadRevisions();
    }
    async function loadBlocks() {
      const c = ctx();
      const res = await fetch(`/v1/memory/blocks?app_name=${c.app}&user_id=${c.user}`, { headers: headers() });
      const data = await res.json();
      const blocks = data.blocks || [];
      document.getElementById("blocksList").innerHTML = blocks.length
        ? blocks.map(b => `<div class="card"><div><span class="badge b-blue">${escapeHtml(b.label)}</span> <span class="muted">updated ${b.updated_at} by ${b.updated_by}</span></div><pre>${escapeHtml(b.value)}</pre></div>`).join("")
        : '<div class="card muted">No working memory blocks defined yet.</div>';
    }
    async function saveBlock() {
      const c = ctx();
      await fetch("/v1/memory/blocks", {
        method: "POST", headers: headers(),
        body: JSON.stringify({
          app_name: c.rawApp, user_id: c.rawUser,
          label: document.getElementById("blkLabel").value.trim(),
          value: document.getElementById("blkValue").value,
          mode: document.getElementById("blkMode").value,
          actor: "console_ui"
        })
      });
      loadBlocks(); loadRevisions();
    }
    async function loadRevisions() {
      const c = ctx();
      const res = await fetch(`/v1/memory/revisions?app_name=${c.app}&user_id=${c.user}`, { headers: headers() });
      const data = await res.json();
      const revs = data.revisions || [];
      document.getElementById("revisionsList").innerHTML = revs.length
        ? revs.map(r => `<div class="card"><div><span class="badge b-amber">${r.action}</span> <strong>${r.memory_id}</strong> <span class="muted">by ${r.actor} · ${r.timestamp}</span></div><div class="muted">${escapeHtml(r.reason || "")}</div>${r.previous_content ? `<div class="muted">Before: ${escapeHtml(r.previous_content)}</div>` : ""}${r.new_content ? `<div>After: ${escapeHtml(r.new_content)}</div>` : ""}</div>`).join("")
        : '<div class="card muted">No memory revisions recorded yet.</div>';
    }
    async function loadOntology() {
      const [pRes, aRes] = await Promise.all([
        fetch("/v1/domains/packs", { headers: headers() }),
        fetch("/v1/ontology/audit", { headers: headers() }),
      ]);
      const pData = await pRes.json();
      const aData = await aRes.json();
      const packs = Object.values(pData.packs || {});
      document.getElementById("packsList").innerHTML = packs.length
        ? packs.map(p => `<div class="card row"><div><strong>${escapeHtml(p.pack_id)}</strong> ${p.certified ? '<span class="badge b-green">CERTIFIED</span>' : '<span class="badge b-amber">DRAFT</span>'}<div class="muted">Classes: ${Object.keys(p.classes||{}).length} · Relations: ${Object.keys(p.relations||{}).length}</div></div>${!p.certified ? `<button onclick="certifyPack('${p.pack_id}')">Certify Pack</button>` : ''}</div>`).join("")
        : '<div class="card muted">No Domain Packs registered.</div>';
      document.getElementById("auditLog").textContent = JSON.stringify(aData.entries || [], null, 2);
    }
    async function certifyPack(packId) {
      await fetch("/v1/domains/packs/certify", {
        method: "POST", headers: headers(),
        body: JSON.stringify({ pack_id: packId, certified_by: "console_steward" })
      });
      loadOntology();
    }
    async function evolveOntology() {
      const c = ctx();
      await fetch("/v1/ontology/evolve", {
        method: "POST", headers: headers(),
        body: JSON.stringify({ app_name: c.rawApp, min_support_count: 2 })
      });
      loadOntology();
    }
    async function loadMetrics() {
      const res = await fetch("/v1/metrics", { headers: headers() });
      const data = await res.json();
      const counters = data.counters || {};
      document.getElementById("metricsCards").innerHTML = Object.entries(counters).map(([k, v]) =>
        `<div class="card"><div class="muted">${k}</div><div class="kpi">${v}</div></div>`
      ).join("") || '<div class="card muted">No metrics recorded yet.</div>';
      document.getElementById("metricsRaw").textContent = JSON.stringify(data, null, 2);
    }
    function escapeHtml(str) {
      return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }
    function refreshAll() { loadMemories(); loadBlocks(); loadRevisions(); loadOntology(); loadMetrics(); }
    refreshAll();
  </script>
</body>
</html>
"""
