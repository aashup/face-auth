/**
 * Zero-dependency mock backend for the example app.
 *
 * Endpoints (all JSON, Bearer token accepted but not enforced):
 *   POST /enroll      { personnelId, embedding, name? }  → stores a template (base64 f32)
 *   POST /provision   { attestation, since }              → full roster + thresholds + manifest
 *   POST /attendance  { events: [...] }                   → { acked: [ids] } (idempotent)
 *   GET  /roster                                          → { enrolled: [{ id, name, templates }] }
 *   GET  /health                                          → { ok, roster, version }
 *
 * Run:  node server/index.js   (listens on :8080)
 * Device:  adb reverse tcp:8080 tcp:8080   then set awsSyncUrl = http://localhost:8080
 *
 * NOTE: real templates must be produced by the SAME embedding model as the
 * device. This mock simply stores whatever the device enrolls (POST /enroll),
 * then serves it back via /provision — enough to exercise the full
 * provision → identify → attendance → sync → purge loop offline.
 */
const http   = require('http');
const fs     = require('fs');
const path   = require('path');

const PORT = process.env.PORT || 8080;

// ── In-memory state ────────────────────────────────────────────────────────────
const roster      = new Map(); // personnelId -> string[] (base64 f32 embeddings)
const names       = new Map(); // personnelId -> display name
let   version     = 0;         // bumps on every enroll; used as the delta cursor
const ackedEvents = new Set(); // dedupe set for attendance events

// ── Helpers ────────────────────────────────────────────────────────────────────
function loadManifest() {
  try {
    const p = path.join(__dirname, '..', 'models', 'manifest.json');
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch {
    return undefined;
  }
}

function send(res, code, body) {
  const json = JSON.stringify(body);
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(json);
}

function readBody(req) {
  return new Promise((resolve) => {
    let data = '';
    req.on('data', (c) => (data += c));
    req.on('end', () => {
      try   { resolve(data ? JSON.parse(data) : {}); }
      catch { resolve({}); }
    });
  });
}

// ── Request router ─────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  const body = await readBody(req);
  const url  = req.url || '';

  // ── POST /enroll ─────────────────────────────────────────────────────────────
  if (req.method === 'POST' && url.endsWith('/enroll')) {
    const { personnelId, embedding, name } = body;
    if (!personnelId || !embedding) {
      return send(res, 400, { error: 'personnelId + embedding required' });
    }
    const displayName = (name || personnelId).trim();
    const list = roster.get(personnelId) || [];
    list.push(embedding);
    roster.set(personnelId, list);
    names.set(personnelId, displayName);
    version += 1;
    console.log(`[enroll] "${displayName}" id=${personnelId} — ${list.length} template(s), v${version}`);
    return send(res, 200, { ok: true, personnelId, name: displayName, count: list.length });
  }

  // ── POST /provision ───────────────────────────────────────────────────────────
  if (req.method === 'POST' && url.endsWith('/provision')) {
    const since     = Number(body.since || 0);
    const manifest  = loadManifest();
    const thresholds = { matchCosine: 0.6, spoofReject: 0.5 };
    // Always reply full for simplicity; cursor = current version.
    const templates = [];
    for (const [personnelId, list] of roster.entries()) {
      for (const embedding of list) {
        templates.push({ personnelId, embedding });
      }
    }
    console.log(`[provision] since=${since} → ${templates.length} template(s) for ${roster.size} person(s) (v${version})`);
    return send(res, 200, { mode: 'full', templates, thresholds, manifest, syncedAt: version || Date.now() });
  }

  // ── POST /attendance ──────────────────────────────────────────────────────────
  if (req.method === 'POST' && url.endsWith('/attendance')) {
    const events = Array.isArray(body.events) ? body.events : [];
    const acked  = [];
    for (const e of events) {
      if (e && e.id && !ackedEvents.has(e.id)) ackedEvents.add(e.id);
      if (e && e.id) acked.push(e.id); // idempotent: ack again on retry
    }
    console.log(`[attendance] received ${events.length}, acked ${acked.length}`);
    return send(res, 200, { acked });
  }

  // ── GET /roster ───────────────────────────────────────────────────────────────
  if (req.method === 'GET' && url.endsWith('/roster')) {
    const enrolled = [];
    let totalTemplates = 0;
    for (const [id, list] of roster.entries()) {
      enrolled.push({ id, name: names.get(id) ?? id, templates: list.length });
      totalTemplates += list.length;
    }
    return send(res, 200, { enrolled, totalTemplates, version });
  }

  // ── GET /health ───────────────────────────────────────────────────────────────
  if (req.method === 'GET' && url.endsWith('/health')) {
    return send(res, 200, { ok: true, roster: roster.size, version });
  }

  send(res, 404, { error: 'not found' });
});

server.listen(PORT, () => {
  console.log(`\nmock face-auth backend  →  http://localhost:${PORT}`);
  console.log('  adb reverse tcp:8080 tcp:8080  then awsSyncUrl=http://localhost:8080');
  console.log('  GET /roster   — list enrolled people');
  console.log('  GET /health   — server status\n');
});
