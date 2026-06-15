#!/usr/bin/env python3
"""Read-only web dashboard for the TPU request-manager 'apply' progress.

Single-file, stdlib-only (no FastAPI/uvicorn dependency) so it runs under any
python3 on the box. It only READS the manager's own output files and shells out
to `request_manager.py status` for the authoritative per-type plan. It never
writes to or mutates manager state.

Endpoints:
  GET /            -> HTML page (auto-refreshes every 15s via fetch)
  GET /api/state   -> JSON snapshot

Launch:
  python3 dashboard.py --host 0.0.0.0 --port 8092
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MGR_DIR = os.path.dirname(BASE_DIR)  # webui/ lives inside the manager dir
STATE = os.path.join(MGR_DIR, "request_state.json")
EVENTS = os.path.join(MGR_DIR, "events.jsonl")
DEMAND = os.path.join(MGR_DIR, "request_demand.yaml")
MGR_PY = os.path.join(MGR_DIR, "request_manager.py")
# audit cache lives in the sibling tpu_dls/ dir (same one request_manager reads)
AUDIT_CACHE = os.path.join(os.path.dirname(MGR_DIR), "tpu_dls", ".tpu_audit_records.json")

LOOP_INTERVAL = 600          # only used for the "alive" heuristic
STATUS_TTL = 20.0            # cache `status` output this long
EVENT_TAIL = 60             # recent events to surface

_status_lock = threading.Lock()
_status_cache = {"ts": 0.0, "data": None}


# --------------------------------------------------------------------------- #
# data helpers (all read-only)
# --------------------------------------------------------------------------- #
def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def classify(text):
    """Map a gcloud error tail to a coarse blocker class."""
    t = text or ""
    if "RATE_LIMIT" in t or "rate_limit" in t or "CreateRequestsPerMinute" in t:
        return "rate_limit"
    if "no more capacity" in t or '"code": 8' in t:
        return "capacity"
    if "Quota" in t or "RESOURCE_EXHAUSTED" in t:
        return "quota"
    if not t:
        return "ok"
    return "error"


def tail_events(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, 300_000)
            f.seek(size - block)
            data = f.read().decode("utf-8", "replace")
    except Exception:
        return []
    lines = [l for l in data.splitlines() if l.strip()]
    out = []
    for l in lines[-n:]:
        try:
            out.append(json.loads(l))
        except Exception:
            continue
    out.reverse()  # newest first
    return out


def slim_event(e):
    ok = e.get("ok")
    return {
        "ts_iso": e.get("ts_iso") or "",
        "ts": e.get("ts"),
        "kind": e.get("kind") or "",
        "type": e.get("tpu_type") or "",
        "zone": e.get("zone") or "",
        "ok": ok,
        "reason": e.get("error_class") or e.get("skip_reason")
        or ("" if ok else "error"),
        "name": e.get("name") or "",
    }


_DEMAND_RE = re.compile(
    r"^\s*(\S+):\s+target=(\d+)\s+idle=(\d+)\s+pending_new=(\d+)"
    r"\s+effective=(\d+)\s+deficit=(\d+)\s+zones=(.+)$"
)


def parse_status(text):
    out = {
        "demands": [],
        "planned_creates": [],
        "planned_deletes": 0,
        "cache_age": None,
        "records_in_scope": None,
        "enabled": None,
        "dry_run": None,
        "raw_ok": bool(text),
    }
    for line in text.splitlines():
        m = _DEMAND_RE.match(line)
        if m:
            out["demands"].append({
                "type": m.group(1),
                "target": int(m.group(2)),
                "idle": int(m.group(3)),
                "pending_new": int(m.group(4)),
                "effective": int(m.group(5)),
                "deficit": int(m.group(6)),
                "zones": m.group(7).strip(),
            })
            continue
        mc = re.match(r"\s*create (\S+) in (\S+)", line)
        if mc:
            out["planned_creates"].append({"type": mc.group(1), "zone": mc.group(2)})
            continue
        md = re.search(r"Planned deletes:\s*(\d+)", line)
        if md:
            out["planned_deletes"] = int(md.group(1))
            continue
        me = re.search(r"enabled=(\S+)\s+dry_run=(\S+)", line)
        if me:
            out["enabled"] = me.group(1) == "True"
            out["dry_run"] = me.group(2) == "True"
            continue
        ma = re.search(r"cache_age_seconds=(\d+)\s+records_in_scope=(\d+)", line)
        if ma:
            out["cache_age"] = int(ma.group(1))
            out["records_in_scope"] = int(ma.group(2))
    return out


def get_status():
    now = time.time()
    with _status_lock:
        c = _status_cache
        if c["data"] is not None and now - c["ts"] < STATUS_TTL:
            return c["data"]
        try:
            r = subprocess.run(
                [sys.executable, MGR_PY, "status"],
                cwd=MGR_DIR, capture_output=True, text=True, timeout=90,
            )
            text = r.stdout or ""
        except Exception:
            text = ""
        data = parse_status(text)
        c["ts"] = now
        c["data"] = data
        return data


_TYPE_RE = re.compile(r"(v\d+[a-z]?-\d+)")


def fleet_overview():
    """Current full-fleet snapshot from the audit cache, grouped by type x status.

    The audit records carry no tpu_type field, so it is derived from the VM name.
    Reflects the live fleet (every owner), not just the manager's demand types.
    """
    d = read_json(AUDIT_CACHE)
    recs = d.get("records") or []
    ts = d.get("ts")
    agg = {}
    for r in recs:
        m = _TYPE_RE.search(r.get("name") or "")
        t = m.group(1) if m else "?"
        status = (r.get("status") or "?").upper()
        a = agg.setdefault(t, {"IDLE": 0, "BUSY": 0, "other": 0, "total": 0})
        if status == "IDLE":
            a["IDLE"] += 1
        elif status == "BUSY":
            a["BUSY"] += 1
        else:
            a["other"] += 1
        a["total"] += 1
    types = [dict(type=t, **v) for t, v in agg.items()]
    types.sort(key=lambda x: (-x["total"], x["type"]))
    return {
        "ts": ts,
        "age": (time.time() - ts) if ts else None,
        "total": sum(x["total"] for x in types),
        "idle": sum(x["IDLE"] for x in types),
        "busy": sum(x["BUSY"] for x in types),
        "types": types,
    }


def build_state():
    st = read_json(STATE)
    status = get_status()

    managed = st.get("managed_vms", {})
    by_type = {}
    for v in managed.values():
        t = v.get("tpu_type", "?")
        by_type[t] = by_type.get(t, 0) + 1

    blockers = []
    for key, cd in (st.get("cooldowns", {}) or {}).items():
        fails = cd.get("consecutive_failures", 0)
        if fails <= 0:
            continue
        typ, _, zone = key.partition("|")
        blockers.append({
            "type": typ, "zone": zone, "fails": fails,
            "reason": classify(cd.get("last_output_tail", "")),
        })
    blockers.sort(key=lambda b: b["fails"], reverse=True)

    last_loop = st.get("last_loop", {}) or {}
    age = (time.time() - last_loop["ts"]) if last_loop.get("ts") else None

    return {
        "now": time.time(),
        "alive": age is not None and age < 3 * LOOP_INTERVAL,
        "last_loop": last_loop,
        "last_loop_age": age,
        "status": status,
        "managed_by_type": by_type,
        "managed_total": len(managed),
        "blockers": blockers[:24],
        "events": [slim_event(e) for e in tail_events(EVENTS, EVENT_TAIL)],
        "fleet": fleet_overview(),
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>TPU Request Manager · apply 进度</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light">
<style>
:root{
  --bg-grad-1:#f6f4ef; --bg-grad-2:#eceae3;
  --surface:#ffffff; --surface-2:#f6f5f1; --surface-3:#eceae3; --surface-hover:#f1efe9;
  --border:#e2ded4; --border-soft:#ece9e1; --border-strong:#d2cdbf;
  --text:#1c1e23; --text-soft:#515663; --text-dim:#767b88; --text-faint:#9aa0ac;
  --accent:#a87a1e; --accent-strong:#936a16; --accent-soft:#f0e4c8;
  --blue:#3f74c4; --blue-soft:#dbe7f7; --blue-ink:#2a5fa3;
  --red:#c25738; --red-soft:#f6e0d8; --red-ink:#c33049; --red-bg:#fbe2e3;
  --purple:#7a6aa6; --purple-soft:#ece4f4; --purple-ink:#5f5186;
  --green:#3f917a; --green-soft:#dcebe4; --green-ink:#2f6e5c;
  --gray:#8b94a3; --gray-soft:#e8e6df; --gray-ink:#6a6f7c;
  --shadow-card:0 1px 2px rgba(40,36,28,.05),0 6px 18px rgba(40,36,28,.07);
  --shadow-pop:0 6px 18px rgba(40,36,28,.13);
  --radius:13px; --radius-sm:9px; --pill:999px; --bar-h:8px;
  --font:"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans SC",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --font-num:ui-rounded,"SF Pro Rounded",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --maxw:1180px; --ease:cubic-bezier(.22,.61,.36,1);
}
*{box-sizing:border-box}
body{margin:0;font-family:var(--font);color:var(--text);line-height:1.5;
  background:linear-gradient(180deg,var(--bg-grad-1),var(--bg-grad-2));background-attachment:fixed;
  min-height:100vh;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.num{font-family:var(--font-num);font-variant-numeric:tabular-nums}

/* header */
.site-header{position:sticky;top:0;z-index:30;border-bottom:1px solid var(--border);
  background:color-mix(in srgb,var(--bg-grad-1) 86%,transparent);
  -webkit-backdrop-filter:saturate(120%) blur(12px);backdrop-filter:saturate(120%) blur(12px)}
.header-inner{max-width:var(--maxw);margin:0 auto;padding:15px clamp(14px,3vw,28px) 13px;
  display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap}
.brand{display:flex;gap:13px;align-items:center;min-width:0}
.brand-logo{flex:none;width:38px;height:38px;border-radius:11px;display:grid;place-items:center;
  background:var(--accent-soft);color:var(--accent-strong)}
.brand-logo svg{width:21px;height:21px;display:block}
.brand-title{margin:0;font-size:clamp(17px,2vw,21px);font-weight:750;letter-spacing:-.2px;color:var(--text)}
.brand-subtitle{margin:3px 0 0;font-size:12.5px;line-height:1.5;color:var(--text-dim)}
.brand-subtitle strong{color:var(--text-soft);font-weight:650}
.header-meta{display:flex;align-items:center;gap:10px;flex:none;flex-wrap:wrap}
.live{display:inline-flex;align-items:center;gap:7px;padding:6px 12px;border-radius:var(--pill);
  font-size:12.5px;font-weight:700;background:var(--green-soft);color:var(--green-ink)}
.live.dead{background:var(--red-bg);color:var(--red-ink)}
.live .ldot{width:8px;height:8px;border-radius:50%;background:currentColor}
.live:not(.dead) .ldot{animation:pulse 1.6s var(--ease) infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.5)}}
.updated{font-size:12.5px;color:var(--text-dim);white-space:nowrap}
.btn-refresh{display:inline-flex;align-items:center;gap:7px;font:inherit;font-size:13px;font-weight:700;
  color:var(--text);background:var(--surface-2);border:1px solid var(--border);border-radius:var(--pill);
  padding:7px 14px;cursor:pointer;transition:background .18s var(--ease),border-color .18s var(--ease),transform .1s}
.btn-refresh:hover{background:var(--surface-hover);border-color:var(--border-strong)}
.btn-refresh:active{transform:scale(.96)}
.btn-refresh .ic{display:inline-block;font-size:15px;line-height:1}
.btn-refresh.is-loading{pointer-events:none;opacity:.7}
.btn-refresh.is-loading .ic{animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

main{max-width:var(--maxw);margin:0 auto;padding:22px clamp(14px,3vw,28px) 60px}

/* stat cards */
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:20px}
@media(max-width:880px){.stats{grid-template-columns:repeat(3,1fr)}}
@media(max-width:520px){.stats{grid-template-columns:repeat(2,1fr)}}
.scard{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow-card);padding:13px 15px}
.scard .lab{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:700;color:var(--text-soft);letter-spacing:.2px}
.scard .sdot{width:7px;height:7px;border-radius:2px;background:var(--accent);flex:none}
.scard.g .sdot{background:var(--green)}.scard.a .sdot{background:var(--accent)}.scard.r .sdot{background:var(--red)}
.scard.b .sdot{background:var(--blue)}.scard.p .sdot{background:var(--purple)}
.scard .val{font-family:var(--font-num);font-size:23px;font-weight:800;margin-top:7px;letter-spacing:-.4px;
  font-variant-numeric:tabular-nums;line-height:1.1}
.scard .val small{font-size:12.5px;color:var(--text-dim);font-weight:600;font-family:var(--font)}

/* cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow-card);overflow:hidden;margin-bottom:18px}
.card-head{display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:14px 17px;border-bottom:1px solid var(--border-soft)}
.card-title{margin:0;display:flex;align-items:center;gap:8px;font-size:14px;font-weight:800;color:var(--text);letter-spacing:.2px}
.card-title .tdot{width:7px;height:7px;border-radius:2px;background:var(--accent);flex:none}
.card-hint{font-size:12.5px;color:var(--text-dim);font-weight:600}
.cols{display:grid;grid-template-columns:1.12fr .88fr;gap:18px}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.pad{padding:15px 17px}
.empty{padding:22px 17px;color:var(--text-dim);font-size:13px;text-align:center}

/* demand rows */
.drow{padding:13px 17px;border-bottom:1px solid var(--border-soft)}
.drow:last-child{border-bottom:none}
.drow-top{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:9px}
.drow-name{font-size:14px;font-weight:800;color:var(--text);min-width:0;letter-spacing:.2px}
.drow-zones{font-weight:600;font-size:11.5px;color:var(--text-faint);margin-left:8px}
.drow-val{flex:none;display:inline-flex;align-items:baseline;gap:8px}
.drow-val .n{font-family:var(--font-num);font-size:14.5px;font-weight:800;font-variant-numeric:tabular-nums}
.drow-val .n small{color:var(--text-dim);font-weight:700;font-size:12px}
.bar{position:relative;height:var(--bar-h);background:var(--surface-3);border-radius:var(--pill);overflow:hidden}
.bar-fill{position:absolute;inset:0 auto 0 0;width:0;border-radius:var(--pill);background:var(--accent);transition:width .65s var(--ease)}
.bar-fill::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.28));opacity:.5}
.badge{font-size:11px;font-weight:800;letter-spacing:.3px;padding:3px 9px;border-radius:var(--pill);white-space:nowrap;line-height:1.4}
.badge.def{background:var(--accent-soft);color:var(--accent-strong)}
.badge.full{background:var(--green-soft);color:var(--green-ink)}

/* fleet overview */
.frow{padding:11px 17px;border-bottom:1px solid var(--border-soft)}
.frow:last-child{border-bottom:none}
.frow-top{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:8px}
.frow-name{font-size:13.5px;font-weight:800;letter-spacing:.2px}
.ftag{font-size:10px;font-weight:800;padding:2px 7px;border-radius:var(--pill);background:var(--accent-soft);color:var(--accent-strong);margin-left:7px;letter-spacing:.3px}
.frow-val{font-family:var(--font-num);font-size:12.5px;font-weight:700;color:var(--text-dim);font-variant-numeric:tabular-nums;flex:none}
.frow-val .i{color:var(--green-ink);font-weight:800}.frow-val .b{color:var(--blue-ink);font-weight:800}.frow-val .t{color:var(--text);font-weight:800}
.fbar{display:flex;height:7px;border-radius:var(--pill);overflow:hidden;background:var(--surface-3)}
.fbar .si{background:var(--green)}.fbar .sb{background:var(--blue)}.fbar .so{background:var(--gray)}

/* blockers */
.brow{display:flex;align-items:center;gap:11px;padding:11px 17px;border-bottom:1px solid var(--border-soft)}
.brow:last-child{border-bottom:none}
.bkey{flex:1;min-width:0;font-size:13px;font-weight:700;color:var(--text)}
.bkey .z{color:var(--text-dim);font-weight:600}
.bfail{font-family:var(--font-num);font-size:13px;font-weight:800;color:var(--red);font-variant-numeric:tabular-nums}
.pill{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:var(--pill);
  font-size:11px;font-weight:800;letter-spacing:.2px;white-space:nowrap}
.pill::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.pill.quota{background:var(--accent-soft);color:var(--accent-strong)}
.pill.capacity{background:var(--blue-soft);color:var(--blue-ink)}
.pill.rate_limit{background:var(--purple-soft);color:var(--purple-ink)}
.pill.error{background:var(--red-bg);color:var(--red-ink)}
.pill.ok{background:var(--green-soft);color:var(--green-ink)}

/* managed pool */
.stack{display:flex;height:16px;border-radius:var(--radius-sm);overflow:hidden;margin-bottom:14px;background:var(--surface-3)}
.legend{display:flex;flex-direction:column;gap:10px}
.lrow{display:flex;align-items:center;gap:10px;font-size:13px}
.ldotc{width:11px;height:11px;border-radius:3px;flex:none}
.lname{font-weight:700;letter-spacing:.2px}
.lbar{flex:1;max-width:150px;height:6px;border-radius:var(--pill);background:var(--surface-3);overflow:hidden}
.lbar i{display:block;height:100%;border-radius:var(--pill)}
.lcount{margin-left:auto;font-family:var(--font-num);font-weight:800;font-variant-numeric:tabular-nums}

/* events */
.events{max-height:470px;overflow:auto}
.erow{display:grid;grid-template-columns:84px 64px 84px 1fr auto;align-items:center;gap:12px;
  padding:9px 17px 9px 14px;border-bottom:1px solid var(--border-soft);border-left:3px solid transparent}
.erow:last-child{border-bottom:none}
.erow.ok{border-left-color:var(--green)}
.erow.bad{border-left-color:var(--red);background:rgba(194,87,56,.035)}
.etime{font-size:11.5px;color:var(--text-dim);font-variant-numeric:tabular-nums}
.ekind{font-size:11px;font-weight:800;letter-spacing:.3px}
.ekind.create{color:var(--blue-ink)}.ekind.delete{color:var(--purple-ink)}
.etype{font-weight:800;font-size:13px;letter-spacing:.2px}
.ezone{font-size:12px;color:var(--text-faint);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.eright{display:flex;align-items:center;gap:9px;justify-content:flex-end}
.eres{font-size:12px;font-weight:800}.eres.ok{color:var(--green-ink)}.eres.bad{color:var(--red-ink)}
.events::-webkit-scrollbar{width:9px}
.events::-webkit-scrollbar-thumb{background:var(--border-strong);border-radius:9px}

.foot{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:8px;
  font-size:11.5px;color:var(--text-faint)}
.foot code{font-family:ui-monospace,Menlo,monospace;color:var(--text-dim)}
</style></head><body>

<header class="site-header"><div class="header-inner">
  <div class="brand">
    <span class="brand-logo" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round">
        <rect x="3" y="4" width="18" height="6" rx="1.7"/><rect x="3" y="14" width="18" height="6" rx="1.7"/>
        <line x1="6.4" y1="7" x2="6.4" y2="7"/><line x1="6.4" y1="17" x2="6.4" y2="17"/>
      </svg>
    </span>
    <div class="brand-text">
      <h1 class="brand-title">TPU Request Manager</h1>
      <p class="brand-subtitle">申卡 <strong>autoscaler &amp; reaper</strong> · 实时 apply 进度</p>
    </div>
  </div>
  <div class="header-meta">
    <span class="live" id="live"><span class="ldot"></span><span id="live-txt">连接中</span></span>
    <span class="updated" id="updated">更新于 —</span>
    <button class="btn-refresh" id="refresh" type="button" aria-label="刷新"><span class="ic">↻</span><span>刷新</span></button>
  </div>
</div></header>

<main>
  <div class="stats" id="stats"></div>

  <section class="card">
    <div class="card-head"><h2 class="card-title"><span class="tdot"></span>需求 vs 供给</h2><span class="card-hint" id="demand-hint"></span></div>
    <div id="demands"></div>
  </section>

  <section class="card">
    <div class="card-head"><h2 class="card-title"><span class="tdot" style="background:var(--blue)"></span>全集群概览 · all types</h2><span class="card-hint" id="fleet-hint"></span></div>
    <div id="fleet"></div>
  </section>

  <div class="cols">
    <section class="card">
      <div class="card-head"><h2 class="card-title">创建受阻 · blockers</h2><span class="card-hint" id="blockers-hint"></span></div>
      <div id="blockers"></div>
    </section>
    <section class="card">
      <div class="card-head"><h2 class="card-title">托管池 · managed</h2><span class="card-hint" id="managed-hint"></span></div>
      <div class="pad" id="managed"></div>
    </section>
  </div>

  <section class="card">
    <div class="card-head"><h2 class="card-title">近期事件</h2><span class="card-hint" id="events-hint"></span></div>
    <div class="events" id="events"></div>
  </section>

  <div class="foot">
    <span>只读 · 读取 <code>request_state.json</code> / <code>events.jsonl</code> / <code>request_manager.py status</code></span>
    <span id="servertime"></span>
  </div>
</main>

<script>
const PAL=["#3f74c4","#3f917a","#7a6aa6","#a87a1e","#c25738","#5f7494","#8b94a3"];
const colorCache={};
function colorFor(t,allTypes){if(colorCache[t])return colorCache[t];
  const i=allTypes.indexOf(t);colorCache[t]=PAL[(i<0?Object.keys(colorCache).length:i)%PAL.length];return colorCache[t];}
function esc(x){return String(x==null?"":x).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function ago(s){if(s==null)return"?";s=Math.max(0,Math.round(s));
  if(s<60)return s+"秒前";if(s<3600)return Math.floor(s/60)+"分钟前";
  if(s<86400)return Math.floor(s/3600)+"小时前";return Math.floor(s/86400)+"天前";}
function agoEvt(e){let t=e.ts?e.ts*1000:Date.parse(e.ts_iso);if(isNaN(t))return"";return ago((Date.now()-t)/1000);}
function pill(k){return k?`<span class="pill ${esc(k)}">${esc(k)}</span>`:"";}

async function load(manual){
  const btn=document.getElementById("refresh");
  if(manual)btn.classList.add("is-loading");
  let d;
  try{d=await(await fetch("/api/state",{cache:"no-store"})).json();}
  catch(e){const l=document.getElementById("live");l.className="live dead";
    document.getElementById("live-txt").textContent="无法连接";btn.classList.remove("is-loading");return;}
  const s=d.status||{}, ll=d.last_loop||{};

  const live=document.getElementById("live");
  live.className="live"+(d.alive?"":" dead");
  document.getElementById("live-txt").textContent=(d.alive?"loop 存活":"loop 停滞")+" · "+ago(d.last_loop_age);

  const pc=(s.planned_creates&&s.planned_creates.length!=null)?s.planned_creates.length:(s.planned_creates||0);
  const totalDef=(s.demands||[]).reduce((a,x)=>a+(x.deficit||0),0);
  const stats=[
    {c:totalDef>0?"a":"g",lab:"总缺口",val:totalDef===0?'0 <small>已满足</small>':totalDef+' <small>张待补</small>'},
    {c:"b",lab:"本轮",val:`<span style="color:var(--green-ink)">+${ll.create_ok??0}</span> <small>建</small> <span style="color:var(--purple-ink)">−${ll.delete_ok??0}</span> <small>删</small>`},
    {c:"p",lab:"计划创建",val:`${pc} <small>个</small>`},
    {c:"g",lab:"托管 VM",val:d.managed_total??"–"},
    {c:s.cache_age!=null&&s.cache_age<180?"g":"a",lab:"审计缓存",val:(s.cache_age!=null?s.cache_age+'<small>秒</small>':"–")},
    {c:s.dry_run?"a":"g",lab:"模式",val:s.dry_run?'<small style="color:var(--accent-strong)">DRY-RUN</small>':'<small style="color:var(--green-ink)">EXECUTE</small>'},
  ];
  document.getElementById("stats").innerHTML=stats.map(x=>
    `<div class="scard ${x.c}"><div class="lab"><span class="sdot"></span>${x.lab}</div><div class="val">${x.val}</div></div>`).join("");

  document.getElementById("demand-hint").textContent=(s.demands||[]).length+" 型号";
  const dem=(s.demands||[]).map(x=>{
    const pct=x.target>0?Math.min(100,Math.round(x.effective/x.target*100)):(x.effective>0?100:0);
    const col=x.deficit<=0?"var(--green)":(x.effective>0?"var(--accent)":"var(--red)");
    const badge=x.deficit>0?`<span class="badge def">−${x.deficit}</span>`:`<span class="badge full">已满</span>`;
    return `<div class="drow"><div class="drow-top">
        <span class="drow-name">${esc(x.type)}<span class="drow-zones">${esc((x.zones||"").replace(/,/g," · "))}</span></span>
        <span class="drow-val"><span class="n">${x.effective}<small>/${x.target}</small></span>${badge}</span></div>
      <div class="bar"><div class="bar-fill" style="width:${pct}%;background:${col}"></div></div></div>`;}).join("");
  document.getElementById("demands").innerHTML=dem||'<div class="empty">status 不可用 — 无法运行 request_manager.py status</div>';

  const fl=d.fleet||{}; const demSet=new Set((s.demands||[]).map(x=>x.type));
  document.getElementById("fleet-hint").textContent=(fl.total!=null?fl.total+" 台 · "+(fl.idle||0)+" 闲 / "+(fl.busy||0)+" 忙":"")+(fl.age!=null?" · 缓存 "+Math.round(fl.age)+"秒":"");
  const fr=(fl.types||[]).map(x=>{const other=x.other||0,tot=x.total||1;
    const tag=demSet.has(x.type)?'<span class="ftag">暖池</span>':"";
    return `<div class="frow"><div class="frow-top">
        <span class="frow-name">${esc(x.type)}${tag}</span>
        <span class="frow-val"><span class="i">${x.IDLE} 闲</span> · <span class="b">${x.BUSY} 忙</span>${other?" · "+other+" 其它":""} · <span class="t">${x.total}</span></span></div>
      <div class="fbar"><div class="si" style="width:${x.IDLE/tot*100}%"></div><div class="sb" style="width:${x.BUSY/tot*100}%"></div><div class="so" style="width:${other/tot*100}%"></div></div></div>`;}).join("");
  document.getElementById("fleet").innerHTML=fr||'<div class="empty">audit 缓存不可用</div>';

  document.getElementById("blockers-hint").textContent=(d.blockers||[]).length?(d.blockers.length+" 个活跃"):"";
  const bl=(d.blockers||[]).map(x=>`<div class="brow">
      <span class="bkey">${esc(x.type)} <span class="z">@ ${esc(x.zone)}</span></span>
      ${pill(x.reason)}<span class="bfail">×${x.fails}</span></div>`).join("");
  document.getElementById("blockers").innerHTML=bl||'<div class="empty">无活跃冷却 — 创建未被阻塞</div>';

  const mt=Object.entries(d.managed_by_type||{}).sort((a,b)=>b[1]-a[1]);
  const tot=mt.reduce((a,x)=>a+x[1],0)||1; const allT=mt.map(x=>x[0]);
  document.getElementById("managed-hint").textContent=tot+" 台";
  const seg=mt.map(([t,n])=>`<div style="width:${n/tot*100}%;background:${colorFor(t,allT)}" title="${esc(t)}: ${n}"></div>`).join("");
  const leg=mt.map(([t,n])=>`<div class="lrow"><span class="ldotc" style="background:${colorFor(t,allT)}"></span>
      <span class="lname">${esc(t)}</span>
      <span class="lbar"><i style="width:${n/tot*100}%;background:${colorFor(t,allT)}"></i></span>
      <span class="lcount">${n}</span></div>`).join("");
  document.getElementById("managed").innerHTML=mt.length?(`<div class="stack">${seg}</div><div class="legend">${leg}</div>`):'<div class="empty">无托管 VM</div>';

  const ev=d.events||[];
  document.getElementById("events-hint").textContent=ev.length?("最近 "+ev.length):"";
  document.getElementById("events").innerHTML=ev.length?ev.map(x=>`<div class="erow ${x.ok?"ok":"bad"}">
      <span class="etime" title="${esc(x.ts_iso)}">${agoEvt(x)}</span>
      <span class="ekind ${esc(x.kind)}">${esc(x.kind)}</span>
      <span class="etype">${esc(x.type)}</span>
      <span class="ezone" title="${esc(x.name)}">${esc(x.zone)}</span>
      <span class="eright">${x.reason?pill(x.reason):""}<span class="eres ${x.ok?"ok":"bad"}">${x.ok?"✓ 成功":"✕ 失败"}</span></span>
    </div>`).join(""):'<div class="empty">暂无事件</div>';

  document.getElementById("updated").textContent="更新于 "+new Date().toLocaleTimeString("zh-CN",{hour12:false});
  document.getElementById("servertime").innerHTML="服务器 "+esc(new Date(d.now*1000).toISOString().replace("T"," ").slice(0,19))+" UTC";
  btn.classList.remove("is-loading");
}
document.getElementById("refresh").addEventListener("click",()=>load(true));
load();
setInterval(load,15000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            try:
                body = json.dumps(build_state())
            except Exception as exc:  # never 500 the page
                body = json.dumps({"error": str(exc)})
            self._send(200, body, "application/json")
        elif path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/healthz":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, *_):  # silence access log
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"apply-dashboard on http://{args.host}:{args.port}  (mgr_dir={MGR_DIR})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
