const state = {
  view: "library",
  documents: [],
  selectedDocument: null,
  extractionKey: null,
  extractionVersion: localStorage.getItem("corpus-lens-extraction-version") || "2.0",
  category: "",
  sort: "newest",
  search: "",
  inspectorTab: "overview",
  atlasKind: "entity",
  atlasSearch: "",
  pipeline: null,
  pipelineStage: null,
  pipelineTab: null,
  contexts: null,
  contextMode: "profiles",
  contextSelection: "global",
  contextVersion: null,
  contextCategory: null,
  contextFilter: "all",
  effectiveProfileConfig: {},
  instructionPreview: null,
  instructionWorker: "extractor",
  instructionPreviewRequest: 0,
  promptInstructionSource: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHtml = (value = "") => String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const formatNumber = value => new Intl.NumberFormat("en-US", { notation: Number(value) > 9999 ? "compact" : "standard" }).format(Number(value || 0));
const formatDate = value => value ? new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" }).format(new Date(value)) : "Unknown date";
const formatDuration = seconds => `${Math.floor((seconds || 0) / 60)} min`;
const formatRuntime = seconds => Number(seconds || 0) >= 3600 ? `${(Number(seconds) / 3600).toFixed(1)} hr` : formatDuration(seconds);
const formatTime = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || payload.error || "Request failed");
  return payload;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 1800);
}

function errorHtml(error) {
  return `<div class="error-block"><strong>Could not load this view.</strong><br>${escapeHtml(error.message)}</div>`;
}

function setView(view) {
  state.view = view;
  $$(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === view));
  $$(".view").forEach(node => node.classList.toggle("active", node.id === `view-${view}`));
  $(".sidebar").classList.remove("open");
  history.replaceState(null, "", `#${view}`);
  if (view === "atlas") loadAtlas();
  if (view === "quality") loadQuality();
  if (view === "pipeline") loadPipeline();
  if (view === "contexts") loadContexts();
}

async function bootstrap() {
  bindEvents();
  try {
    const [overview, facets] = await Promise.all([api("/api/overview"), api("/api/facets")]);
    $("#corpus-stat").textContent = `${formatNumber(overview.totals.documents)} documents · ${formatNumber(overview.totals.words)} words`;
    renderCategories(facets.categories);
    await loadDocuments();
    const target = location.hash.replace("#", "");
    if (["library", "atlas", "quality", "pipeline", "contexts"].includes(target)) setView(target);
  } catch (error) {
    $("#document-list").innerHTML = errorHtml(error);
  }
}

function bindEvents() {
  $("#global-version").value = state.extractionVersion;
  $("#global-version").addEventListener("change", event => {
    state.extractionVersion = event.target.value;
    localStorage.setItem("corpus-lens-extraction-version", state.extractionVersion);
    if (state.selectedDocument) {
      const selected = state.selectedDocument.extraction_history?.find(item => item.version === state.extractionVersion);
      state.extractionKey = selected?.key || state.selectedDocument.active_extraction_key;
      renderInspector();
    }
    $("#quality-table").removeAttribute("data-loaded");
    loadDocuments();
    if (state.view === "atlas") loadAtlas();
    if (state.view === "quality") loadQuality();
    if (state.view === "pipeline") {
      state.pipeline = null;
      state.pipelineStage = null;
      state.pipelineTab = null;
      loadPipeline();
    }
    toast(`Using extraction V${state.extractionVersion}`);
  });
  $$(".nav-item").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
  $$("[data-context-mode]").forEach(button => button.addEventListener("click", () => {
    state.contextMode = button.dataset.contextMode;
    state.contextFilter = "all";
    renderContexts();
  }));
  $(".mobile-menu").addEventListener("click", () => $(".sidebar").classList.toggle("open"));
  $("#document-sort").addEventListener("change", event => { state.sort = event.target.value; loadDocuments(); });
  $("#global-search").addEventListener("input", debounce(event => {
    state.search = event.target.value.trim();
    if (state.view === "atlas") { state.atlasSearch = state.search; $("#atlas-search").value = state.search; loadAtlas(); }
    else { setView("library"); loadDocuments(); }
  }, 220));
  $("#atlas-search").addEventListener("input", debounce(event => { state.atlasSearch = event.target.value.trim(); loadAtlas(); }, 200));
  $$("#atlas-kinds button").forEach(button => button.addEventListener("click", () => {
    state.atlasKind = button.dataset.kind;
    $$("#atlas-kinds button").forEach(item => item.classList.toggle("active", item === button));
    loadAtlas();
  }));
  document.addEventListener("keydown", event => {
    if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
      event.preventDefault(); $("#global-search").focus();
    }
    if (event.key === "Escape") { $(".sidebar").classList.remove("open"); $("#inspector").classList.remove("open"); }
  });
}

function renderCategories(categories) {
  $("#category-filter").innerHTML = [
    `<button class="active" data-category="">All</button>`,
    ...categories.map(item => `<button data-category="${item.value}">${escapeHtml(item.label)} · ${item.count}</button>`)
  ].join("");
  $$("#category-filter button").forEach(button => button.addEventListener("click", () => {
    state.category = button.dataset.category;
    $$("#category-filter button").forEach(item => item.classList.toggle("active", item === button));
    loadDocuments();
  }));
}

async function loadDocuments() {
  const params = new URLSearchParams({ sort: state.sort, version: state.extractionVersion });
  if (state.search) params.set("search", state.search);
  if (state.category) params.set("category", state.category);
  try {
    const payload = await api(`/api/documents?${params}`);
    state.documents = payload.documents;
    $("#document-count").textContent = `${payload.count} document${payload.count === 1 ? "" : "s"}`;
    renderDocuments();
  } catch (error) { $("#document-list").innerHTML = errorHtml(error); }
}

function renderDocuments() {
  const list = $("#document-list");
  if (!state.documents.length) { list.innerHTML = `<div class="loading-block">No documents match this view.</div>`; return; }
  list.innerHTML = state.documents.map(doc => `
    <button class="document-card ${state.selectedDocument?.media_id === doc.media_id ? "active" : ""}" data-media-id="${escapeHtml(doc.media_id)}">
      <img class="thumb" src="${escapeHtml(doc.thumbnail_url || "")}" alt="" loading="lazy">
      <span><strong>${escapeHtml(doc.title)}</strong><small>${escapeHtml(doc.category_name || "Uncategorized")} · ${formatDate(doc.publish_date)} ${doc.latest_extraction === "v2" ? `<span class="doc-version">V${escapeHtml(doc.extraction_version)} latest</span>` : ""}</small>
      <span class="doc-counts"><span>${formatNumber(doc.word_count)} words</span><span>${doc.entity_count} entities</span><span>${doc.reference_count} refs</span></span></span>
    </button>`).join("");
  $$(".document-card", list).forEach(card => card.addEventListener("click", () => openDocument(card.dataset.mediaId)));
}

async function openDocument(mediaId) {
  $("#transcript-pane").innerHTML = `<div class="loading-block">Opening transcript…</div>`;
  $(".library-layout").classList.add("document-open");
  try {
    state.selectedDocument = await api(`/api/documents/${encodeURIComponent(mediaId)}`);
    state.inspectorTab = "overview";
    state.extractionKey = state.selectedDocument.extraction_history?.find(item => item.version === state.extractionVersion)?.key || state.selectedDocument.active_extraction_key || state.selectedDocument.extraction_history?.[0]?.key || null;
    renderDocuments();
    renderTranscript();
    renderInspector();
    setView("library");
  } catch (error) { $("#transcript-pane").innerHTML = errorHtml(error); }
}

function renderTranscript() {
  const doc = state.selectedDocument;
  const segments = (doc.content?.timestamped_segments || []).filter(segment => segment.text?.trim());
  const transcript = segments.length ? segments : [{ text: doc.content?.full_text || "", start_seconds: 0 }];
  $("#transcript-pane").innerHTML = `
    <div class="doc-hero">
      <img class="doc-hero-bg" src="${escapeHtml(doc.thumbnail_url || "")}" alt="">
      <div class="hero-actions"><button class="hero-action" id="mobile-back" title="Back to library" aria-label="Back to library">‹</button><a class="hero-action" href="${escapeHtml(doc.url)}" target="_blank" rel="noreferrer" title="Open source video" aria-label="Open source video">↗</a><button class="hero-action" id="open-inspector" title="Open extractions" aria-label="Open extractions">✦</button></div>
      <span class="eyebrow">${escapeHtml(doc.category_name || "Uncategorized")}</span>
      <h1>${escapeHtml(doc.title)}</h1>
      <div class="doc-meta"><span>${escapeHtml(doc.channel_name || "Ad hoc source")}</span><span>${formatDate(doc.publish_date)}</span><span>${formatDuration(doc.duration)}</span><span>${formatNumber(doc.views)} views</span></div>
    </div>
    <div class="transcript-toolbar"><strong>Transcript</strong><input class="transcript-search" type="search" placeholder="Find in transcript…"><span class="segment-count">${formatNumber(doc.word_count)} words · ${formatNumber(transcript.length)} segments</span></div>
    <div class="transcript-body">${transcript.map(segment => `<div class="segment" data-text="${escapeHtml(segment.text.toLowerCase())}" data-original="${escapeHtml(segment.text)}"><button class="timestamp" data-time="${segment.start_seconds || 0}" title="Open video at this timestamp">${formatTime(segment.start_seconds || 0)}</button><div class="segment-text">${escapeHtml(segment.text)}</div></div>`).join("")}</div>`;
  $(".transcript-search").addEventListener("input", event => filterTranscript(event.target.value));
  $$(".timestamp").forEach(button => button.addEventListener("click", () => window.open(`${doc.url}${doc.url.includes("?") ? "&" : "?"}t=${Math.floor(button.dataset.time)}s`, "_blank", "noopener")));
  $("#mobile-back").addEventListener("click", () => $(".library-layout").classList.remove("document-open"));
  $("#open-inspector").addEventListener("click", () => $("#inspector").classList.add("open"));
}

function filterTranscript(rawQuery) {
  const query = rawQuery.trim().toLowerCase();
  let visible = 0;
  $$(".segment").forEach(segment => {
    const show = !query || segment.dataset.text.includes(query);
    segment.classList.toggle("hidden", !show);
    if (show) {
      visible += 1;
      const original = segment.dataset.original;
      $(".segment-text", segment).innerHTML = query ? escapeHtml(original).replace(new RegExp(escapeRegExp(query), "gi"), match => `<mark>${match}</mark>`) : escapeHtml(original);
    }
  });
  if (query) toast(`${visible} matching transcript segment${visible === 1 ? "" : "s"}`);
}

function renderInspector() {
  const doc = state.selectedDocument;
  const snapshot = currentExtraction(doc);
  const x = snapshot.data;
  const counts = { overview: "Overview", entities: `Entities ${x.entities.length}`, references: `References ${x.references.length}` };
  $("#inspector").innerHTML = `
    <div class="inspector-head">
      <div class="inspector-title-row"><div><span class="eyebrow">Extraction</span><h2>Evidence map</h2></div>${versionPicker(doc, snapshot)}</div>
      <div class="inspector-tabs">${Object.entries(counts).map(([key, label]) => `<button class="${state.inspectorTab === key ? "active" : ""}" data-tab="${key}">${label}</button>`).join("")}</div>
    </div>
    <div class="inspector-body">${inspectorContent(doc, snapshot)}</div>`;
  $("#extraction-version")?.addEventListener("change", event => {
    state.extractionKey = event.target.value;
    renderInspector();
  });
  $$(".inspector-tabs button").forEach(button => button.addEventListener("click", () => { state.inspectorTab = button.dataset.tab; renderInspector(); }));
  $$("[data-highlight]").forEach(button => button.addEventListener("click", () => {
    const query = button.dataset.highlight;
    $(".transcript-search").value = query;
    filterTranscript(query);
    $("#transcript-pane").scrollTo({ top: 220, behavior: "smooth" });
    $("#inspector").classList.remove("open");
  }));
}

function currentExtraction(doc) {
  const history = doc.extraction_history || [];
  return history.find(item => item.key === state.extractionKey) || history[0] || {
    key: "v1",
    version: doc.extraction_version || "1.0",
    label: `V${doc.extraction_version || "1.0"}`,
    status: "production",
    run_mode: "production",
    completed_at: doc.content?.metadata_updated_at,
    profile_version_ids: [],
    data: doc.extractions,
  };
}

function versionPicker(doc, selected) {
  const history = doc.extraction_history || [selected];
  return `<label class="version-picker"><span>Version</span><select id="extraction-version" aria-label="Extraction version">${history.map(item => `<option value="${escapeHtml(item.key)}" ${item.key === selected.key ? "selected" : ""}>${escapeHtml(item.label)}</option>`).join("")}</select></label>`;
}

function inspectorContent(doc, snapshot) {
  const x = snapshot.data;
  if (state.inspectorTab === "entities") return extractionCards(x.entities, "entity");
  if (state.inspectorTab === "references") return extractionCards(x.references, "reference");
  return `
    <div class="fact-strip"><div><strong>${x.topics.length}</strong><small>Topics</small></div><div><strong>${x.entities.length}</strong><small>Entities</small></div><div><strong>${x.references.length}</strong><small>References</small></div></div>
    <div class="category-block"><small>Category</small><strong>${escapeHtml(x.category || doc.category_name || "Not assigned")}</strong></div>
    <div class="section-label">Topics</div><div class="tag-cloud">${tagButtons(x.topics)}</div>
    <div class="section-label">Participants</div>${participantRows(x.participants)}
    <div class="section-label">Keywords</div><div class="tag-cloud">${tagButtons(x.keywords)}</div>
    <div class="section-label">Extraction record</div>${extractionRecord(doc, snapshot)}`;
}

function extractionCards(items, kind) {
  if (!items.length) return `<div class="loading-block">No ${kind} records were extracted for this document.</div>`;
  return items.map(item => `<div class="extraction-card"><div class="extraction-card-head"><span class="type-swatch ${kind}"></span><button class="tag" data-highlight="${escapeHtml(item.raw_label || item.label)}">${escapeHtml(item.label)}</button><span class="type">${escapeHtml(item.type)}</span></div>${item.context ? `<p class="extraction-context">${escapeHtml(item.context)}</p>` : ""}${item.evidence ? `<blockquote class="evidence-quote">${escapeHtml(item.evidence)}</blockquote>` : ""}<div class="extraction-meta"><span class="usage">${escapeHtml(item.usage)}${item.mention_count ? ` · ${item.mention_count} mention${item.mention_count === 1 ? "" : "s"}` : ""}</span>${resolutionBadge(item)}</div></div>`).join("");
}

function participantRows(participants) {
  if (!participants.length) return `<p class="extraction-context">No participants extracted.</p>`;
  return participants.map(person => `<div class="participant"><span class="avatar">${escapeHtml(initials(person.name))}</span><span><strong>${escapeHtml(person.name)}</strong><small>${escapeHtml(person.role || "Role not extracted")}</small>${person.raw_name && person.raw_name !== person.name ? `<small>From “${escapeHtml(person.raw_name)}”</small>` : ""}</span>${resolutionBadge(person)}</div>${person.evidence ? `<blockquote class="evidence-quote participant-evidence">${escapeHtml(person.evidence)}</blockquote>` : ""}`).join("");
}

function resolutionBadge(item) {
  if (!item.resolution_status) return "";
  const confidence = item.confidence == null ? "" : ` · ${Math.round(Number(item.confidence) * 100)}%`;
  return `<span class="resolution-badge ${escapeHtml(item.resolution_status)}" title="${escapeHtml(item.resolution_basis || "Resolution status")}">${escapeHtml(item.resolution_status)}${confidence}</span>`;
}

function extractionRecord(doc, snapshot) {
  const profiles = snapshot.profile_version_ids?.length ? snapshot.profile_version_ids.map(value => `v${value}`).join(", ") : "Production record";
  const details = snapshot.run_id
    ? `${escapeHtml(snapshot.run_mode || "run")} · ${escapeHtml(snapshot.model || "model unavailable")} · profiles ${escapeHtml(profiles)}`
    : `Metadata updated ${formatDate(doc.content.metadata_updated_at)} · quality ${doc.extraction_quality_score ?? "not scored"}`;
  return `<div class="extraction-card record-card"><div class="extraction-card-head"><span class="type-swatch"></span><strong>${escapeHtml(snapshot.label)}</strong><span class="release-badge ${escapeHtml(snapshot.status)}">${escapeHtml(snapshot.status)}</span></div><p class="extraction-context">${details}</p><span class="usage">Completed ${formatDate(snapshot.completed_at)}</span></div>${runUsageBreakdown(snapshot)}`;
}

function runUsageBreakdown(snapshot) {
  if (!snapshot.run_id) return `<div class="run-usage unavailable"><span class="section-label">Run usage</span><p>Token usage was not recorded for this production record.</p></div>`;
  const usage = snapshot.usage || {};
  const tracked = usage.tracking_status === "recorded";
  const totalTokens = tracked ? formatNumber(usage.total_tokens) : "Not recorded";
  const totalCost = tracked ? formatCost(usage.estimated_cost) : "Not recorded";
  const elapsed = snapshot.duration_seconds == null ? "Unknown" : `${Number(snapshot.duration_seconds).toFixed(1)}s`;
  return `<div class="run-usage ${tracked ? "recorded" : "unavailable"}">
    <span class="section-label">Run usage</span>
    <div class="usage-totals"><div><strong>${totalTokens}</strong><small>Total tokens</small></div><div><strong>${totalCost}</strong><small>Estimated cost</small></div><div><strong>${formatNumber(snapshot.input_characters)}</strong><small>Transcript chars</small></div><div><strong>${elapsed}</strong><small>Elapsed</small></div></div>
    <details class="usage-details"><summary><span>Step breakdown</span><small>${snapshot.chunk_count || 0} transcript chunks</small></summary>
      <div class="usage-step-header"><span>Step</span><span>Input</span><span>Output</span><span>Total</span><span>Cost</span></div>
      ${(usage.steps || []).map(usageStepRow).join("")}
      <p class="usage-basis">${escapeHtml(usage.pricing_basis || "Usage metadata unavailable.")}</p>
    </details>
  </div>`;
}

function usageStepRow(step) {
  const tokenValue = value => value == null ? "—" : formatNumber(value);
  return `<div class="usage-step"><span><strong>${escapeHtml(step.label || step.step)}</strong><small>${escapeHtml(step.model || step.kind || "")}${step.request_count == null ? "" : ` · ${step.request_count} req`}</small></span><span>${tokenValue(step.input_tokens)}</span><span>${tokenValue(step.output_tokens)}</span><span>${tokenValue(step.total_tokens)}</span><span>${step.estimated_cost == null ? "—" : formatCost(step.estimated_cost)}</span></div>`;
}

function formatCost(value) {
  if (value == null) return "—";
  const amount = Number(value);
  if (amount > 0 && amount < 0.0001) return "<$0.0001";
  return `$${amount.toFixed(4)}`;
}

function tagButtons(items) { return items.length ? items.map(item => `<button class="tag" data-highlight="${escapeHtml(item)}">${escapeHtml(item)}</button>`).join("") : `<span class="extraction-context">Nothing extracted</span>`; }
function initials(name) { return name.split(/\s+/).slice(0, 2).map(part => part[0]).join("").toUpperCase(); }

async function loadAtlas() {
  const index = $("#atlas-index");
  index.innerHTML = `<div class="loading-block">Building the ${state.atlasKind} index…</div>`;
  $("#atlas-detail").innerHTML = `<div class="empty-state compact"><span class="empty-kicker">Cross-document view</span><h2>Select an item from the index.</h2></div>`;
  const params = new URLSearchParams({ kind: state.atlasKind, version: state.extractionVersion });
  if (state.atlasSearch) params.set("search", state.atlasSearch);
  try {
    const payload = await api(`/api/atlas?${params}`);
    index.innerHTML = payload.items.length ? payload.items.map((item, i) => `
      <button class="atlas-row" data-key="${escapeHtml(item.key)}"><span class="atlas-rank">${String(i + 1).padStart(2, "0")}</span><span><strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(item.type)} · ${formatNumber(item.mention_count)} mentions</small></span><span class="atlas-count">${item.document_count}</span></button>`).join("") : `<div class="loading-block">No matching relationships.</div>`;
    $$(".atlas-row", index).forEach(row => row.addEventListener("click", () => loadAtlasDetail(row.dataset.key, row)));
  } catch (error) { index.innerHTML = errorHtml(error); }
}

async function loadAtlasDetail(key, row) {
  $$(".atlas-row").forEach(item => item.classList.toggle("active", item === row || item.dataset.key === String(key)));
  const detail = $("#atlas-detail");
  detail.innerHTML = `<div class="loading-block">Tracing connections…</div>`;
  try {
    const payload = await api(`/api/atlas/documents?kind=${encodeURIComponent(state.atlasKind)}&key=${encodeURIComponent(key)}&version=${encodeURIComponent(state.extractionVersion)}`);
    const item = payload.item || { label: key, type: state.atlasKind };
    detail.innerHTML = `<div class="atlas-title"><span class="eyebrow">${escapeHtml(item.type)}</span><h2>${escapeHtml(item.label)}</h2><p>${payload.documents.length} connected document${payload.documents.length === 1 ? "" : "s"} in the current corpus</p></div>${payload.profile ? collectionProfile(payload.profile) : ""}<div class="linked-list"><div class="section-label">Documents in this view</div>${payload.documents.map(doc => `<article class="linked-doc" data-media-id="${escapeHtml(doc.media_id)}"><img src="${escapeHtml(doc.thumbnail_url || "")}" alt="" loading="lazy"><div><h3>${escapeHtml(doc.title)}</h3><div class="linked-meta">${escapeHtml(doc.category_name || "Uncategorized")} · ${formatDate(doc.publish_date)} · ${escapeHtml(doc.usage)}</div>${doc.context ? `<p>${escapeHtml(doc.context)}</p>` : ""}</div></article>`).join("")}</div>`;
    $$(".linked-doc", detail).forEach(doc => doc.addEventListener("click", () => openDocument(doc.dataset.mediaId)));
    $$(".profile-link", detail).forEach(link => link.addEventListener("click", () => jumpAtlas(link.dataset.kind, link.dataset.key)));
  } catch (error) { detail.innerHTML = errorHtml(error); }
}

function collectionProfile(profile) {
  const stats = profile.stats || {};
  return `<section class="collection-profile">
    <div class="collection-stats"><div><strong>${formatNumber(stats.documents)}</strong><small>Documents</small></div><div><strong>${formatNumber(stats.words)}</strong><small>Words</small></div><div><strong>${formatNumber(stats.views)}</strong><small>Views</small></div><div><strong>${formatRuntime(stats.duration_seconds)}</strong><small>Runtime</small></div></div>
    <div class="signature-grid">${signatureGroup("Topics", "topic", profile.topics)}${signatureGroup("Entities", "entity", profile.entities)}${signatureGroup("References", "reference", profile.references)}</div>
  </section>`;
}

function signatureGroup(label, kind, items) {
  return `<div class="signature-group"><span class="section-label">Top ${label}</span>${items.length ? items.map(item => `<button class="profile-link" data-kind="${kind}" data-key="${escapeHtml(item.key)}"><span>${escapeHtml(item.label)}</span><strong>${item.count}</strong></button>`).join("") : `<small>No ${label.toLowerCase()} extracted</small>`}</div>`;
}

function jumpAtlas(kind, key) {
  state.atlasKind = kind;
  $$("#atlas-kinds button").forEach(button => button.classList.toggle("active", button.dataset.kind === kind));
  loadAtlas();
  loadAtlasDetail(key, null);
}

async function loadQuality() {
  if ($("#quality-table").dataset.loaded === state.extractionVersion) return;
  try {
    const payload = await api(`/api/quality?version=${encodeURIComponent(state.extractionVersion)}`);
    $("#coverage-grid").innerHTML = payload.coverage.map(item => `<div class="coverage-card"><strong>${item.percent}%</strong><span>${escapeHtml(item.label)} · ${item.covered}/${item.total}</span><div class="meter"><i style="width:${item.percent}%"></i></div></div>`).join("");
    $("#quality-count").textContent = `${payload.documents.length} transcripts assessed`;
    $("#quality-table").dataset.loaded = state.extractionVersion;
    $("#quality-table").innerHTML = `<div class="quality-row header"><span>Document</span><span>Cat.</span><span>Topics</span><span>Keys</span><span>People</span><span>Entities</span><span>Refs</span><span>Coverage</span></div>${payload.documents.map(doc => `<div class="quality-row"><div class="quality-doc" data-media-id="${escapeHtml(doc.media_id)}"><img src="${escapeHtml(doc.thumbnail_url || "")}" alt=""><strong>${escapeHtml(doc.title)}</strong></div>${qualityValue(doc.category_name ? 1 : 0)}${qualityValue(doc.topics)}${qualityValue(doc.keywords)}${qualityValue(doc.participants)}${qualityValue(doc.entities)}${qualityValue(doc.references)}<div class="score ${doc.coverage_score < 60 ? "low" : doc.coverage_score < 90 ? "medium" : ""}"><span class="score-dot"></span>${doc.coverage_score}%</div></div>`).join("")}`;
    $$(".quality-doc").forEach(doc => doc.addEventListener("click", () => openDocument(doc.dataset.mediaId)));
  } catch (error) { $("#quality-table").innerHTML = errorHtml(error); }
}

function qualityValue(value) { return `<div class="quality-value ${value ? "" : "zero"}">${value || "—"}</div>`; }

async function loadContexts() {
  if (state.contexts) { renderContexts(); return; }
  try {
    state.contexts = await api("/api/extraction/v2/contexts");
    renderContexts();
  } catch (error) {
    $("#context-detail").innerHTML = errorHtml(error);
  }
}

function renderContexts() {
  if (!state.contexts) return;
  $$("[data-context-mode]").forEach(button => button.classList.toggle("active", button.dataset.contextMode === state.contextMode));
  $("#alias-count").textContent = formatNumber(state.contexts.counts.aliases);
  $("#learning-count").textContent = formatNumber(state.contexts.counts.learning_events);
  $("#context-layout").classList.toggle("context-list-mode", state.contextMode !== "profiles");
  if (state.contextMode === "aliases") renderResolutionMemory();
  else if (state.contextMode === "learning") renderLearningProposals();
  else renderProfileHierarchy();
}

function contextGroups(data) {
  const totalDocuments = data.sources.reduce((sum, item) => sum + Number(item.document_count || 0), 0);
  const channelCategories = data.profiles.filter(profile => profile.scope === "channel_category").map(profile => ({
    key: `channel_category:${profile.channel_id}:${profile.category_id}`,
    scope: "channel_category",
    name: `${profile.channel_name || profile.channel_id} · ${profile.category_name || profile.category_id}`,
    channel_id: profile.channel_id,
    source_id: profile.source_id,
    category_id: profile.category_id,
    document_count: 0,
  }));
  return [
    { label: "Global", nodes: [{ key: "global", scope: "global", name: "Global baseline", document_count: totalDocuments }] },
    { label: "Sources", nodes: data.sources.map(item => ({ ...item, key: `source:${item.source_id}`, scope: "source", name: item.source_name })) },
    { label: "Categories", nodes: data.categories.map(item => ({ ...item, key: `category:${item.category_id}`, scope: "category", name: item.category_name })) },
    { label: "Channels", nodes: data.channels.map(item => ({ ...item, key: `channel:${item.channel_id}`, scope: "channel", name: item.channel_name })) },
    { label: "Channel + category", nodes: channelCategories },
  ];
}

function renderProfileHierarchy() {
  const data = state.contexts;
  const groups = contextGroups(data);
  const nodes = groups.flatMap(group => group.nodes);
  let node = nodes.find(item => item.key === state.contextSelection) || nodes[0];
  state.contextSelection = node.key;
  if (node.scope === "channel" && !state.contextCategory) state.contextCategory = node.categories?.[0]?.category_id || null;
  $("#context-index").innerHTML = `<div class="context-index-head"><span class="eyebrow">Context index</span><strong>${formatNumber(nodes.length)} scopes</strong></div>${groups.map(group => contextIndexGroup(group, data.profiles)).join("")}`;
  $("#context-detail").innerHTML = profileContextDetail(data, node);
  $$('[data-context-key]').forEach(button => button.addEventListener("click", () => {
    state.contextSelection = button.dataset.contextKey;
    state.contextVersion = null;
    state.contextCategory = null;
    renderProfileHierarchy();
  }));
  $("#context-version")?.addEventListener("change", event => { state.contextVersion = Number(event.target.value); renderProfileHierarchy(); });
  $("#context-category")?.addEventListener("change", event => { state.contextCategory = Number(event.target.value); renderProfileHierarchy(); });
  $("[data-open-pipeline]")?.addEventListener("click", () => setView("pipeline"));
  bindInstructionEditor();
  loadProfileInstructionPreview(state.effectiveProfileConfig);
}

function contextIndexGroup(group, profiles) {
  if (!group.nodes.length) return `<section class="context-group"><span>${escapeHtml(group.label)}</span><small>No configured combinations</small></section>`;
  return `<section class="context-group"><span>${escapeHtml(group.label)}</span>${group.nodes.map(node => {
    const configured = profiles.some(profile => profileMatchesNode(profile, node));
    return `<button class="context-node ${state.contextSelection === node.key ? "active" : ""}" data-context-key="${escapeHtml(node.key)}"><i class="${configured ? "configured" : ""}"></i><span><strong>${escapeHtml(node.name)}</strong><small>${formatNumber(node.document_count)} docs · ${configured ? "Configured" : "Inherited"}</small></span></button>`;
  }).join("")}</section>`;
}

function profileMatchesNode(profile, node) {
  if (profile.scope !== node.scope) return false;
  if (node.scope === "global") return true;
  if (node.scope === "source") return profile.source_id === node.source_id;
  if (node.scope === "category") return Number(profile.category_id) === Number(node.category_id);
  if (node.scope === "channel") return profile.channel_id === node.channel_id;
  return profile.channel_id === node.channel_id && Number(profile.category_id) === Number(node.category_id);
}

function activeProfileVersion(profile) {
  if (!profile) return null;
  return profile.versions.find(version => version.status === "active") || profile.versions[0] || null;
}

function profileContextDetail(data, node) {
  const categoryId = node.scope === "channel" ? state.contextCategory : node.category_id;
  const sourceId = node.source_id || (node.scope === "channel" ? node.source_id : null);
  const channelId = node.channel_id || null;
  const slots = [
    { scope: "global", label: "Global", relevant: true, profile: data.profiles.find(profile => profile.scope === "global") },
    { scope: "source", label: "Source", relevant: Boolean(sourceId), profile: data.profiles.find(profile => profile.scope === "source" && profile.source_id === sourceId) },
    { scope: "category", label: "Category", relevant: Boolean(categoryId), profile: data.profiles.find(profile => profile.scope === "category" && Number(profile.category_id) === Number(categoryId)) },
    { scope: "channel", label: "Channel", relevant: Boolean(channelId), profile: data.profiles.find(profile => profile.scope === "channel" && profile.channel_id === channelId) },
    { scope: "channel_category", label: "Channel + category", relevant: Boolean(channelId && categoryId), profile: data.profiles.find(profile => profile.scope === "channel_category" && profile.channel_id === channelId && Number(profile.category_id) === Number(categoryId)) },
  ].filter(slot => slot.relevant);
  const localProfile = data.profiles.find(profile => profileMatchesNode(profile, node));
  const selectedVersion = localProfile ? localProfile.versions.find(version => Number(version.profile_version_id) === Number(state.contextVersion)) || activeProfileVersion(localProfile) : null;
  const applied = slots.filter(slot => slot.profile).map(slot => ({ ...slot, version: slot.profile === localProfile ? selectedVersion : activeProfileVersion(slot.profile) }));
  const effective = {};
  applied.forEach(item => deepMergeConfig(effective, item.version?.config || {}));
  const provenance = buildConfigProvenance(applied);
  state.effectiveProfileConfig = effective;
  state.promptInstructionSource = applied.filter(item => item.version?.config?.prompt_instructions).at(-1) || null;
  const categoryOptions = node.scope === "channel" && node.categories?.length ? `<label class="context-category-picker"><span>Preview category</span><select id="context-category">${node.categories.map(category => `<option value="${category.category_id}" ${Number(category.category_id) === Number(categoryId) ? "selected" : ""}>${escapeHtml(category.category_name)} · ${category.document_count}</option>`).join("")}</select></label>` : "";
  const versionPicker = localProfile ? `<label class="context-version-picker"><span>Local version</span><select id="context-version">${localProfile.versions.map(version => `<option value="${version.profile_version_id}" ${version.profile_version_id === selectedVersion?.profile_version_id ? "selected" : ""}>v${version.version} · ${escapeHtml(version.status)}</option>`).join("")}</select></label>` : "";
  return `<div class="context-detail-head"><div><span class="eyebrow">${escapeHtml(node.scope.replace("_", " + "))} context</span><h2>${escapeHtml(node.name)}</h2><p>${localProfile ? escapeHtml(localProfile.profile_name) : "No local override · effective behavior is inherited"}</p></div><div class="context-head-controls">${categoryOptions}${versionPicker}</div></div>
    <div class="context-metrics"><div><strong>${formatNumber(node.document_count)}</strong><small>Documents</small></div><div><strong>${applied.length}</strong><small>Configured layers</small></div><div><strong>${localProfile?.versions.length || 0}</strong><small>Local versions</small></div><div><strong>${escapeHtml(localProfile?.learning_mode || applied.at(-1)?.profile?.learning_mode || "inherited")}</strong><small>Learning mode</small></div></div>
    <section class="profile-layer-section base-contract-section"><div class="section-line"><div><span class="layer-number">01</span><span class="section-label">Base worker contracts</span></div><button data-open-pipeline>Open global pipeline</button></div><div class="base-worker-contracts" id="base-worker-contracts"><div class="loading-block">Loading runtime contracts…</div></div></section>
    <section class="profile-layer-section inheritance-section"><div class="section-line"><div><span class="layer-number">02</span><span class="section-label">Inherited profile configuration</span></div></div><div class="inheritance-path">${slots.map(slot => inheritanceNode(slot, applied)).join('<i>→</i>')}</div></section>
    <div class="context-config-grid">
      <section class="instruction-stack"><span class="section-label">Instruction overlays</span>${instructionStack(applied, node, localProfile, selectedVersion)}</section>
      <section class="effective-settings"><span class="section-label">Effective settings</span>${settingGroup("Reference modules", effective.reference_modules, provenance, "reference_modules")}${settingGroup("Coverage", effective.coverage, provenance, "coverage")}${settingGroup("Known values", effective.known_values, provenance, "known_values")}${settingGroup("Blocked values", effective.blocked_values, provenance, "blocked_values")}${settingGroup("Model routing", effective.model_routing, provenance, "model_routing")}${settingGroup("Confidence", effective.confidence, provenance, "confidence")}</section>
    </div>
    <section class="profile-layer-section runtime-contract-section"><div class="section-line"><div><span class="layer-number">03</span><span class="section-label">Effective runtime instructions</span></div></div><div id="effective-runtime-contract"><div class="loading-block">Assembling effective worker instructions…</div></div></section>
    ${localProfile ? profileVersionHistory(localProfile, selectedVersion) : ""}
    <details class="effective-json"><summary><span>Effective configuration</span><small>${applied.length} merged profile layer${applied.length === 1 ? "" : "s"}</small></summary><pre><code>${escapeHtml(JSON.stringify(effective, null, 2))}</code></pre></details>`;
}

function inheritanceNode(slot, applied) {
  const selected = applied.find(item => item.profile === slot.profile);
  return `<div class="inheritance-node ${slot.profile ? "configured" : "inherited"}"><span>${escapeHtml(slot.label)}</span><strong>${escapeHtml(slot.profile?.profile_name || "No override")}</strong><small>${selected?.version ? `v${selected.version.version} · ${selected.version.status}` : "Pass through"}</small></div>`;
}

function deepMergeConfig(target, source) {
  Object.entries(source || {}).forEach(([key, value]) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      if (!target[key] || typeof target[key] !== "object" || Array.isArray(target[key])) target[key] = {};
      deepMergeConfig(target[key], value);
    } else target[key] = Array.isArray(value) ? [...value] : value;
  });
  return target;
}

function buildConfigProvenance(applied) {
  const provenance = {};
  const mark = (value, path, source) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      Object.entries(value).forEach(([key, item]) => mark(item, path ? `${path}.${key}` : key, source));
    } else provenance[path] = source;
  };
  applied.forEach(item => {
    const source = `${item.label} v${item.version.version}`;
    Object.entries(item.version.config || {}).forEach(([key, value]) => mark(value, key, source));
  });
  return provenance;
}

function instructionStack(applied, node, localProfile, selectedVersion) {
  const editable = ["category", "channel"].includes(node.scope);
  const inherited = applied.filter(item => item.profile !== localProfile && item.version?.config?.prompt_instructions);
  const localInstructions = selectedVersion?.config?.prompt_instructions;
  const layers = inherited.map(item => `<article class="instruction-layer ${localInstructions ? "overridden" : ""}"><div><span>${escapeHtml(item.label)}</span><small>v${item.version.version} · ${localInstructions ? `overridden by ${node.scope}` : "effective fallback"}</small></div><p>${escapeHtml(item.version.config.prompt_instructions)}</p></article>`);
  if (editable) layers.push(instructionEditor(node, selectedVersion));
  if (!layers.length) return `<div class="context-empty compact"><strong>Base worker contract</strong><p>No scoped instruction overlays are active for this context.</p></div>`;
  return layers.join("");
}

function instructionEditor(node, version) {
  const instructions = version?.config?.prompt_instructions || "";
  const status = version?.status || "new profile";
  return `<form class="instruction-editor" id="context-instruction-form" data-scope="${escapeHtml(node.scope)}" data-category-id="${node.category_id || ""}" data-channel-id="${escapeHtml(node.channel_id || "")}"><div class="instruction-editor-head"><span>${escapeHtml(node.scope)} instruction</span><small>${version ? `v${version.version} · ${escapeHtml(status)}` : escapeHtml(status)}</small></div><textarea id="context-instruction-text" maxlength="20000" aria-label="${escapeHtml(node.scope)} prompt instructions">${escapeHtml(instructions)}</textarea><footer><span><b id="instruction-character-count">${instructions.length.toLocaleString()}</b> / 20,000 characters</span><button type="submit" disabled>Save draft</button></footer></form>`;
}

function bindInstructionEditor() {
  const form = $("#context-instruction-form");
  if (!form) return;
  const textarea = $("#context-instruction-text");
  const button = form.querySelector('button[type="submit"]');
  const initial = textarea.value.trim();
  textarea.addEventListener("input", () => {
    $("#instruction-character-count").textContent = textarea.value.length.toLocaleString();
    button.disabled = textarea.value.trim() === initial;
  });
  form.addEventListener("submit", async event => {
    event.preventDefault();
    button.disabled = true;
    button.textContent = "Saving…";
    try {
      const saved = await api("/api/extraction/v2/profile-instructions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scope: form.dataset.scope,
          category_id: form.dataset.categoryId ? Number(form.dataset.categoryId) : null,
          channel_id: form.dataset.channelId || null,
          prompt_instructions: textarea.value,
        }),
      });
      state.contexts = await api("/api/extraction/v2/contexts");
      state.contextVersion = saved.profile_version_id;
      renderContexts();
      toast(`Draft v${saved.version} saved`);
    } catch (error) {
      button.disabled = false;
      button.textContent = "Save draft";
      toast(error.message || "Could not save draft");
    }
  });
}

async function loadProfileInstructionPreview(config) {
  const requestId = ++state.instructionPreviewRequest;
  state.instructionPreview = null;
  try {
    const preview = await api("/api/extraction/v2/instruction-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    });
    if (requestId !== state.instructionPreviewRequest) return;
    state.instructionPreview = preview;
    renderInstructionPreviewPanels();
  } catch (error) {
    if (requestId !== state.instructionPreviewRequest) return;
    const failure = errorHtml(error);
    if ($("#base-worker-contracts")) $("#base-worker-contracts").innerHTML = failure;
    if ($("#effective-runtime-contract")) $("#effective-runtime-contract").innerHTML = failure;
  }
}

function renderInstructionPreviewPanels() {
  const preview = state.instructionPreview;
  const baseRoot = $("#base-worker-contracts");
  const effectiveRoot = $("#effective-runtime-contract");
  if (!preview || !baseRoot || !effectiveRoot) return;
  baseRoot.innerHTML = preview.workers.map(worker => `<details class="base-worker-card"><summary><span><strong>${escapeHtml(worker.label)}</strong><small>${escapeHtml(worker.role)}</small></span><i>Code contract</i></summary><pre><code>${escapeHtml(worker.base_instructions)}</code></pre></details>`).join("");
  const selected = preview.workers.find(worker => worker.key === state.instructionWorker) || preview.workers[0];
  state.instructionWorker = selected.key;
  const changed = selected.base_instructions !== selected.effective_instructions;
  effectiveRoot.innerHTML = `<div class="runtime-toolbar"><div class="segmented">${preview.workers.map(worker => `<button class="${worker.key === selected.key ? "active" : ""}" data-instruction-worker="${escapeHtml(worker.key)}">${escapeHtml(worker.label)}</button>`).join("")}</div><span class="runtime-change ${changed ? "changed" : ""}">${changed ? "Profile-adjusted" : "Base unchanged"}</span></div><div class="runtime-source"><strong>${escapeHtml(runtimeInstructionSource(selected.key))}</strong><span>${escapeHtml(preview.prompt_instruction_behavior.detail)}</span></div><div class="runtime-inputs"><span>Profile inputs</span>${selected.profile_inputs.map(item => `<code>${escapeHtml(item)}</code>`).join("")}</div><pre class="runtime-code"><code>${escapeHtml(selected.effective_instructions)}</code></pre>`;
  $$('[data-instruction-worker]').forEach(button => button.addEventListener("click", () => { state.instructionWorker = button.dataset.instructionWorker; renderInstructionPreviewPanels(); }));
}

function runtimeInstructionSource(worker) {
  if (worker === "extractor") {
    const source = state.promptInstructionSource;
    return source ? `Scoped prompt supplied by ${source.label} v${source.version.version}` : "No scoped prompt; base Worker 1 contract applies";
  }
  if (worker === "resolver") return "Resolver contract plus effective modules, known values, and runtime aliases";
  return "Learner contract; profile settings control whether and when it runs";
}

function settingGroup(label, value, provenance = {}, path = "") {
  const empty = value == null || (Array.isArray(value) && !value.length) || (typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length);
  if (empty) return `<div class="setting-group"><strong>${escapeHtml(label)}</strong><span class="setting-empty">None configured</span></div>`;
  const source = provenance[path];
  if (Array.isArray(value)) return `<div class="setting-group"><div class="setting-group-head"><strong>${escapeHtml(label)}</strong><small>${escapeHtml(source || "default")}</small></div><div class="setting-chips">${value.map(item => `<span>${escapeHtml(item)}</span>`).join("")}</div></div>`;
  if (typeof value === "object") return `<div class="setting-group"><strong>${escapeHtml(label)}</strong><div class="setting-rows">${Object.entries(value).map(([key, item]) => `<div><span>${escapeHtml(key.replaceAll("_", " "))}</span><small>${escapeHtml(provenance[`${path}.${key}`] || "default")}</small><code>${escapeHtml(Array.isArray(item) ? item.join(", ") : typeof item === "object" ? JSON.stringify(item) : item)}</code></div>`).join("")}</div></div>`;
  return `<div class="setting-group"><div class="setting-group-head"><strong>${escapeHtml(label)}</strong><small>${escapeHtml(source || "default")}</small></div><code>${escapeHtml(value)}</code></div>`;
}

function profileVersionHistory(profile, selectedVersion) {
  return `<section class="profile-history"><span class="section-label">Local version history</span><div>${profile.versions.map(version => `<article class="history-row ${version.profile_version_id === selectedVersion?.profile_version_id ? "selected" : ""}"><span class="release-badge ${escapeHtml(version.status)}">${escapeHtml(version.status)}</span><strong>Version ${version.version}</strong><small>${version.published_at ? `Published ${formatDate(version.published_at)}` : `Created ${formatDate(version.created_at)}`}</small><span>${Object.keys(version.config || {}).length} config fields</span></article>`).join("")}</div></section>`;
}

function renderResolutionMemory() {
  const aliases = state.contexts.aliases;
  const dimensions = ["all", "participant", "entity", "reference", "topic", "keyword"];
  const filtered = state.contextFilter === "all" ? aliases : aliases.filter(item => item.dimension === state.contextFilter);
  $("#context-index").innerHTML = `<div class="context-index-head"><span class="eyebrow">Resolution memory</span><strong>${formatNumber(aliases.length)} aliases</strong></div><section class="memory-filters">${dimensions.map(dimension => `<button class="${state.contextFilter === dimension ? "active" : ""}" data-context-filter="${dimension}"><span>${escapeHtml(dimension === "all" ? "All dimensions" : dimension)}</span><strong>${dimension === "all" ? aliases.length : aliases.filter(item => item.dimension === dimension).length}</strong></button>`).join("")}</section>`;
  $("#context-detail").innerHTML = filtered.length ? `<div class="context-list-head"><div><span class="eyebrow">Resolver knowledge</span><h2>Aliases & canonical values</h2></div><strong>${filtered.length} records</strong></div><div class="memory-table"><div class="memory-row header"><span>Alias</span><span>Canonical value</span><span>Scope</span><span>Trust</span><span>Evidence</span></div>${filtered.map(alias => `<div class="memory-row"><span><strong>${escapeHtml(alias.alias_text)}</strong><small>${escapeHtml(alias.dimension)} · ${escapeHtml(alias.created_by)}</small></span><span>${escapeHtml(alias.canonical_value)}</span><span>${escapeHtml(memoryScope(alias))}</span><span><i class="trust-dot ${escapeHtml(alias.trust_status)}"></i>${escapeHtml(alias.trust_status)}</span><span>${alias.evidence_count}</span></div>`).join("")}</div>` : contextEmpty("Resolution memory is empty", "Trusted aliases will appear here with their scope, evidence, and origin.");
  $$('[data-context-filter]').forEach(button => button.addEventListener("click", () => { state.contextFilter = button.dataset.contextFilter; renderResolutionMemory(); }));
}

function memoryScope(item) {
  return item.channel_name || item.category_name || item.source_name || "Global";
}

function renderLearningProposals() {
  const events = state.contexts.learning_events;
  const statuses = ["all", "proposed", "applied", "rejected", "reversed"];
  const filtered = state.contextFilter === "all" ? events : events.filter(item => item.status === state.contextFilter);
  $("#context-index").innerHTML = `<div class="context-index-head"><span class="eyebrow">Learning queue</span><strong>${formatNumber(events.length)} events</strong></div><section class="memory-filters">${statuses.map(status => `<button class="${state.contextFilter === status ? "active" : ""}" data-context-filter="${status}"><span>${escapeHtml(status === "all" ? "All events" : status)}</span><strong>${status === "all" ? events.length : events.filter(item => item.status === status).length}</strong></button>`).join("")}</section>`;
  $("#context-detail").innerHTML = filtered.length ? `<div class="context-list-head"><div><span class="eyebrow">Worker 3 output</span><h2>Learning proposals</h2></div><strong>${filtered.length} events</strong></div><div class="learning-list">${filtered.map(event => `<article><div><span class="release-badge ${escapeHtml(event.status)}">${escapeHtml(event.status)}</span><strong>${escapeHtml(event.action)} · ${escapeHtml(event.dimension)}</strong><small>${formatDate(event.created_at)} · ${escapeHtml(event.profile_name || "Unscoped")}</small></div><p>${escapeHtml(event.reason)}</p><footer><span>${escapeHtml(event.raw_value || "—")}</span><i>→</i><span>${escapeHtml(event.canonical_value || "—")}</span></footer></article>`).join("")}</div>` : contextEmpty("No learning proposals", "Worker 3 has not stored any profile or alias proposals in the current corpus.");
  $$('[data-context-filter]').forEach(button => button.addEventListener("click", () => { state.contextFilter = button.dataset.contextFilter; renderLearningProposals(); }));
}

function contextEmpty(title, detail) {
  return `<div class="context-empty"><span class="inspector-glyph">◇</span><strong>${escapeHtml(title)}</strong><p>${escapeHtml(detail)}</p></div>`;
}

async function loadPipeline() {
  if (state.pipeline) { renderPipeline(); return; }
  try {
    state.pipeline = await api(`/api/pipeline/prompt?version=${encodeURIComponent(state.extractionVersion)}`);
    renderPipeline();
  } catch (error) {
    $("#pipeline-main").innerHTML = errorHtml(error);
  }
}

function renderPipeline() {
  const data = state.pipeline;
  const selected = data.stages.find(stage => stage.key === state.pipelineStage) || data.stages.find(stage => stage.key === "extractor") || data.stages[0];
  state.pipelineStage = selected.key;
  const views = selected.code_views || [];
  if (!views.some(view => view.key === state.pipelineTab)) state.pipelineTab = views[0]?.key || null;
  $("#stage-rail").innerHTML = `<div class="stage-head"><span class="eyebrow">Execution path · V${escapeHtml(data.version)}</span><strong>${escapeHtml(data.name)}</strong></div>${data.stages.map((stage, index) => `<button class="stage-row ${stage.key === selected.key ? "active" : ""}" data-pipeline-stage="${escapeHtml(stage.key)}"><span class="stage-number">${String(index + 1).padStart(2, "0")}</span><span><strong>${escapeHtml(stage.label)}</strong><small>${escapeHtml(stage.detail)}</small></span></button>`).join("")}`;
  $("#pipeline-main").innerHTML = `
    <div class="contract-bar"><div><span class="eyebrow">Selected stage</span><h2>${escapeHtml(selected.label)}</h2></div><span class="live-badge"><i></i>Live V${escapeHtml(data.version)} path</span></div>
    <div class="contract-stats"><div><small>Owner</small><strong>${escapeHtml(selected.owner)}</strong></div><div><small>Mechanism</small><strong>${escapeHtml(selected.mechanism)}</strong></div><div><small>Invocation</small><strong>${escapeHtml(selected.invocation)}</strong></div><div><small>Prompt schema</small><strong>v${escapeHtml(data.schema_version)}</strong></div></div>
    <div class="stage-summary"><span class="eyebrow">Behind this step</span><p>${escapeHtml(selected.summary)}</p></div>
    <div class="pipeline-workspace">
      <div class="stage-detail">
        <div class="stage-flow">${stageFlowColumn("Inputs", selected.inputs, "input")}${stageFlowColumn("What happens", selected.actions, "action")}${stageFlowColumn("Outputs", selected.outputs, "output")}</div>
        ${views.length ? `<section class="prompt-panel"><div class="prompt-toolbar"><div class="segmented prompt-tabs">${views.map(view => `<button class="${state.pipelineTab === view.key ? "active" : ""}" data-prompt-tab="${escapeHtml(view.key)}">${escapeHtml(view.label)}</button>`).join("")}</div><button class="copy-button" aria-label="Copy current prompt view" title="Copy current view">Copy</button></div><div class="placeholder-note"><strong>Runtime placeholders</strong><span>Values inside {{ braces }} are supplied from the document, active profile, or previous stage.</span></div><pre class="prompt-code"><code>${escapeHtml(stageCodeRaw(selected))}</code></pre></section>` : `<div class="deterministic-note"><strong>No prompt in this stage</strong><span>This step runs through application code or a provider/database client, not an LLM instruction block.</span></div>`}
      </div>
      <aside class="contract-sidebar">
        ${versionLedger(data)}
        <div class="contract-card"><span class="section-label">Category hints · ${data.category_hints.length}</span><div class="hint-list">${data.category_hints.map(hint => `<span>${escapeHtml(hint)}</span>`).join("")}</div></div>
        <div class="contract-card"><span class="section-label">How to read the rail</span><p class="rail-help">Each numbered row is a processing stage, not a version. Select a row to inspect who owns it, whether a model runs, and what crosses the boundary.</p></div>
      </aside>
    </div>`;
  $$('[data-pipeline-stage]').forEach(button => button.addEventListener("click", () => { state.pipelineStage = button.dataset.pipelineStage; state.pipelineTab = null; renderPipeline(); }));
  $$("[data-prompt-tab]").forEach(button => button.addEventListener("click", () => { state.pipelineTab = button.dataset.promptTab; renderPipeline(); }));
  $(".copy-button")?.addEventListener("click", async () => {
    const copied = await copyText(stageCodeRaw(selected));
    toast(copied ? "Copied to clipboard" : "Copy unavailable in this browser");
  });
}

function stageFlowColumn(label, items, kind) {
  return `<section class="flow-column ${kind}"><span class="section-label">${escapeHtml(label)}</span><div>${(items || []).map((item, index) => `<p><i>${String(index + 1).padStart(2, "0")}</i><span>${escapeHtml(item)}</span></p>`).join("")}</div></section>`;
}

function stageCodeRaw(stage) {
  return stage.code_views?.find(view => view.key === state.pipelineTab)?.content || "";
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch (_error) {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    return copied;
  }
}

function versionLedger(data) {
  const registry = data.version_registry.find(item => String(item.version) === String(data.version)) || data.version_registry[0];
  const v2 = data.version === "2.0";
  const applied = (v2 ? data.run_versions : data.media_versions).map(item => v2 ? `${item.document_count} docs · ${item.run_count} runs` : `${item.document_count} docs · v${item.version}`).join(", ");
  return `<div class="contract-card version-card"><span class="section-label">Version ledger</span><div class="version-line"><span>Registry</span><strong>${registry ? `v${escapeHtml(registry.version)}` : "Not registered"}</strong></div><div class="version-line"><span>Rollout</span><strong>${escapeHtml(registry?.rollout_status || "unknown")}</strong></div><div class="version-line"><span>${v2 ? "Completed" : "Applied"}</span><strong>${escapeHtml(applied || "No completed records")}</strong></div><div class="version-line"><span>Output schema</span><strong>v${escapeHtml(data.schema_version)}</strong></div>${registry?.changes_summary ? `<p>${escapeHtml(registry.changes_summary)}</p>` : ""}</div>`;
}

function escapeRegExp(value) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function debounce(fn, wait) { let timer; return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); }; }

bootstrap();
