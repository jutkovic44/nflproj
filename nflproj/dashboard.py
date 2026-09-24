"""Dashboard generator.

Writes one self-contained HTML file per run with the projection data embedded,
so it opens straight from disk with no server and no build step. Every run
overwrites it, which makes the file a live view of the most recent slate.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#FF3C00",
    "DAL": "#041E42", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LA": "#003594", "LAC": "#0080C6", "LV": "#A5ACAF", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#69BE28",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#5A1414",
}


def _players_payload(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        def v(col, nd=1):
            x = r.get(col)
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return round(float(x), nd)
        out.append({
            "name": r["player_display_name"], "team": r["team"], "pos": r["position"],
            "opp": r.get("opponent_team"), "game": r.get("game_key"),
            "slate": r.get("slate"),
            "passYds": v("proj_passing_yards", 0), "passTds": v("proj_passing_tds", 2),
            "passAtt": v("proj_attempts", 0), "ints": v("proj_passing_interceptions", 2),
            "carries": v("proj_carries"), "rushYds": v("proj_rushing_yards", 0),
            "tgts": v("proj_targets"), "rec": v("proj_receptions"),
            "recYds": v("proj_receiving_yards", 0),
            "tdProb": v("proj_anytime_td_prob", 3),
            "questionable": int(r.get("status_questionable") or 0),
            "vacated": v("vacated_target_share", 3),
            "impliedTotal": v("implied_team_total"),
        })
    return out


def build(df: pd.DataFrame, games: pd.DataFrame, meta: dict, path: str = "dashboard.html",
          injuries: dict | None = None, watch: dict | None = None,
          weather: dict | None = None, accuracy: dict | None = None) -> str:
    game_rows = []
    for _, g in games.iterrows():
        home_tot = (g.total_line / 2 + g.spread_line / 2) if pd.notna(g.get("total_line")) else None
        away_tot = (g.total_line - home_tot) if home_tot is not None else None
        game_rows.append({
            "key": f"{g.away_team}@{g.home_team}",
            "away": g.away_team, "home": g.home_team,
            "spread": None if pd.isna(g.get("spread_line")) else float(g.spread_line),
            "total": None if pd.isna(g.get("total_line")) else float(g.total_line),
            "awayTotal": None if away_tot is None else round(float(away_tot), 1),
            "homeTotal": None if home_tot is None else round(float(home_tot), 1),
            "roof": None if pd.isna(g.get("roof")) else str(g.roof),
            "wind": None if pd.isna(g.get("wind")) else float(g.wind),
            "temp": None if pd.isna(g.get("temp")) else float(g.temp),
            "kick": f"{g.weekday} {str(g.gametime)[:5]}",
            "weather": (weather or {}).get(f"{g.away_team}@{g.home_team}"),
            "injuries": {"away": (injuries or {}).get(g.away_team, []),
                         "home": (injuries or {}).get(g.home_team, [])},
            "watch": {"away": (watch or {}).get(g.away_team, []),
                      "home": (watch or {}).get(g.home_team, [])},
        })

    payload = {
        "meta": {**meta, "generated": datetime.now().strftime("%a %b %d, %-I:%M %p")},
        "games": game_rows,
        "players": _players_payload(df),
        "colors": TEAM_COLORS,
        "accuracy": accuracy,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Slate projections</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#0d0f12; --panel:#14181d; --line:#232a32; --fg:#e8ecf1; --dim:#8b97a6;
  --hot:#ff5c39; --cool:#3ddc97; --warn:#f5c518;
  color-scheme: dark;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
*,*::before,*::after{box-sizing:inherit}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0;background:var(--ink);color:var(--fg);
  font-family:'Archivo',system-ui,-apple-system,sans-serif;font-size:15px;line-height:1.45}
.wrap{max-width:1180px;margin:0 auto;padding:22px 16px 80px}
header{border-bottom:2px solid var(--fg);padding-bottom:14px;margin-bottom:18px;
  display:flex;justify-content:space-between;align-items:flex-end;gap:14px;flex-wrap:wrap}
.hl{flex:1;min-width:240px}
#refresh{background:var(--fg);color:var(--ink);border:none;padding:10px 18px;
  font-family:'Archivo',sans-serif;font-weight:800;font-size:13px;text-transform:uppercase;
  letter-spacing:.03em;cursor:pointer;border-radius:2px;white-space:nowrap}
#refresh:hover{background:var(--cool)}
#refresh:disabled{background:var(--line);color:var(--dim);cursor:default}
#refresh.file{background:transparent;border:1px solid var(--line);color:var(--dim)}
#rstat{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--dim);
  margin-top:6px;text-align:right;min-height:15px}
#rstat.err{color:var(--hot)}
h1{font-size:clamp(28px,6vw,46px);font-weight:800;letter-spacing:-.02em;margin:0 0 4px;
   text-transform:uppercase}
.sub{font-family:'IBM Plex Mono',monospace;font-size:12.5px;color:var(--dim)}
.sub b{color:var(--fg);font-weight:500}
.games{display:grid;grid-template-columns:repeat(auto-fill,minmax(228px,1fr));gap:9px;margin:16px 0 22px}
.game{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent,#555);
  padding:10px 12px;cursor:pointer;transition:border-color .12s,background .12s}
.game:hover{background:#1a2027}
.game.on{border-color:var(--fg);background:#1c232b}
.matchup{font-weight:600;font-size:15px;display:flex;justify-content:space-between;align-items:baseline}
.kick{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--dim);font-weight:400}
.totals{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--dim);margin-top:5px;
  display:flex;gap:10px;flex-wrap:wrap}
.totals span b{color:var(--fg);font-weight:500}
.ctl{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px;
  position:sticky;top:env(safe-area-inset-top,0px);background:var(--ink);padding:10px 0;z-index:5;
  border-bottom:1px solid var(--line)}
input[type=search],select{background:var(--panel);border:1px solid var(--line);color:var(--fg);
  padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:12.5px;border-radius:2px}
input[type=search]{flex:1;min-width:150px}
.pills{display:flex;gap:4px}
.pill{background:transparent;border:1px solid var(--line);color:var(--dim);padding:6px 11px;
  font-family:'IBM Plex Mono',monospace;font-size:12px;cursor:pointer;border-radius:2px}
.pill.on{background:var(--fg);color:var(--ink);border-color:var(--fg);font-weight:600}
.tablewrap{overflow-x:auto;border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;min-width:760px}
th{font-family:'IBM Plex Mono',monospace;font-size:10.5px;font-weight:600;color:var(--dim);
  text-align:right;padding:9px 8px;border-bottom:1px solid var(--line);cursor:pointer;
  white-space:nowrap;background:var(--panel);position:sticky;top:0}
th:first-child,th:nth-child(2){text-align:left}
th.sorted{color:var(--fg)}
td{padding:8px;border-bottom:1px solid #1b2128;text-align:right;
  font-family:'IBM Plex Mono',monospace;font-size:13px;font-variant-numeric:tabular-nums}
td.nm{text-align:left;font-family:'Archivo',sans-serif;font-weight:600;font-size:14px;
  white-space:nowrap;border-left:3px solid var(--tc,#444);padding-left:9px}
td.mu{text-align:left;font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--dim);white-space:nowrap}
tr:hover td{background:#171c22}
.q{color:var(--warn);font-size:10px;margin-left:5px;font-family:'IBM Plex Mono',monospace}
.vac{color:var(--cool);font-size:10px;margin-left:5px;font-family:'IBM Plex Mono',monospace}
.tdcell{position:relative;min-width:74px}
.bar{position:absolute;left:0;top:0;bottom:0;background:var(--hot);opacity:.22}
.tdcell span{position:relative}
.muted{color:#4d5763}
.detail{background:var(--panel);border:1px solid var(--line);border-top:none;
  padding:16px 14px;margin:-1px 0 22px;display:none}
.detail.on{display:block}
.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:680px){.dgrid{grid-template-columns:1fr;gap:14px}}
.dcol h4{margin:0 0 9px;font-size:13px;font-weight:800;letter-spacing:.02em;
  padding-bottom:5px;border-bottom:2px solid var(--tcol,#444)}
.wx{font-family:'IBM Plex Mono',monospace;font-size:12.5px;color:var(--fg);
  background:#1a2027;border-left:3px solid var(--cool);padding:9px 11px;margin-bottom:15px}
.wx .imp{display:block;color:var(--warn);margin-top:5px;font-size:11.5px}
.wx .srcs{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.wx .src{font-size:10.5px;color:var(--dim);border:1px solid var(--line);padding:1px 6px}
.conf{font-size:10.5px;padding:1px 6px;margin-left:6px;border:1px solid var(--line);color:var(--dim)}
.conf.good{color:var(--cool);border-color:var(--cool)}
.conf.bad{color:var(--hot);border-color:var(--hot)}
.wx.dome{border-left-color:var(--dim);color:var(--dim)}
.ilist,.wlist{list-style:none;margin:0;padding:0}
.ilist li{display:flex;justify-content:space-between;gap:8px;padding:5px 0;
  border-bottom:1px solid #1b2128;font-size:13px}
.ilist .who{font-weight:600}
.ilist .who small{color:var(--dim);font-weight:400;font-family:'IBM Plex Mono',monospace;
  font-size:10.5px;margin-left:5px}
.ilist .st{font-family:'IBM Plex Mono',monospace;font-size:11px;white-space:nowrap}
.st.Out{color:var(--hot)} .st.Doubtful{color:#ff8a5c} .st.Questionable{color:var(--warn)}
.ilist .inj{color:var(--dim);font-size:11px;font-family:'IBM Plex Mono',monospace}
.sd{font-family:'IBM Plex Mono',monospace;font-size:9.5px;font-style:normal;padding:1px 4px;
  margin-right:6px;border:1px solid var(--line);color:var(--dim)}
.sd.OFF{color:var(--cool);border-color:#245c45}
.sd.DEF{color:#8ab4ff;border-color:#2b3f63}
.wlist li{padding:8px 0;border-bottom:1px solid #1b2128}
.wlist .wn{font-weight:600;font-size:14px}
.wlist .wl{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--fg);margin-top:2px}
.wlist .wy{font-size:11.5px;color:var(--dim);margin-top:2px;line-height:1.35}
.none{color:var(--dim);font-size:12px;font-family:'IBM Plex Mono',monospace}
.score{display:flex;gap:16px;align-items:center;background:var(--panel);
  border:1px solid var(--line);padding:13px 15px;margin:14px 0 0;cursor:pointer;flex-wrap:wrap}
.score:hover{background:#1a2027}
.sbig{font-family:'IBM Plex Mono',monospace;font-size:38px;font-weight:600;line-height:1;
  letter-spacing:-.03em}
.sbig small{font-size:15px;color:var(--dim);font-weight:400}
.smeta{flex:1;min-width:180px}
.sgrade{font-weight:800;font-size:15px;text-transform:uppercase;letter-spacing:.01em}
.snote{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--dim);margin-top:3px}
.topbar{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap}
#refresh{font-family:'IBM Plex Mono',monospace;font-size:12.5px;padding:9px 16px;
  background:var(--fg);color:var(--ink);border:none;cursor:pointer;font-weight:600;
  border-radius:2px;white-space:nowrap}
#refresh:hover{background:var(--cool)}
#refresh:disabled{background:var(--line);color:var(--dim);cursor:default}
#refresh.file{background:transparent;border:1px solid var(--line);color:var(--dim)}
.runlog{display:none;background:#0a0c0f;border:1px solid var(--line);margin:10px 0 0;
  padding:10px 12px;font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--dim);
  max-height:190px;overflow-y:auto;white-space:pre-wrap}
.runlog.on{display:block}
.runlog b{color:var(--cool);font-weight:500}
.spark{display:flex;gap:3px;align-items:flex-end;height:38px}
.spark i{width:9px;background:var(--cool);opacity:.75;display:block;border-radius:1px}
.spark i.lo{background:var(--hot)}
.sdetail{display:none;background:var(--panel);border:1px solid var(--line);border-top:none;
  padding:14px 15px;margin-bottom:6px}
.sdetail.on{display:block}
.comp{margin-bottom:11px}
.comprow{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;
  font-size:12px;margin-bottom:4px}
.track{height:6px;background:#1b2128;position:relative}
.track i{position:absolute;left:0;top:0;bottom:0;background:var(--cool);display:block}
.stat-tb{width:100%;border-collapse:collapse;margin-top:10px;min-width:0}
.stat-tb td,.stat-tb th{font-family:'IBM Plex Mono',monospace;font-size:11.5px;padding:4px 6px;
  border-bottom:1px solid #1b2128;text-align:right;background:none;position:static}
.stat-tb th{color:var(--dim);font-weight:500}
.stat-tb td:first-child,.stat-tb th:first-child{text-align:left}
footer{margin-top:22px;font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--dim);
  border-top:1px solid var(--line);padding-top:12px}
@media (max-width:640px){.wrap{padding:16px 10px 70px}td,th{padding:7px 6px}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="hl">
    <h1 id="title">Slate projections</h1>
    <div class="sub" id="meta"></div>
  </div>
  <div>
    <button id="refresh">Refresh</button>
    <div id="rstat"></div>
  </div>
</header>

<div class="score" id="score"></div>
<div class="sdetail" id="sdetail"></div>

<div class="games" id="games"></div>
<div class="detail" id="detail"></div>

<div class="ctl">
  <input type="search" id="q" placeholder="search player">
  <div class="pills" id="posf"></div>
  <select id="sort"></select>
</div>

<div class="tablewrap">
  <table>
    <thead><tr id="head"></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>

<footer id="foot"></footer>
</div>

<script>
const DATA = __DATA__;
const COLS = [
  {k:'name', t:'Player', kind:'name'},
  {k:'mu',   t:'Matchup', kind:'mu'},
  {k:'passAtt', t:'ATT'}, {k:'passYds', t:'PASS YD'}, {k:'passTds', t:'PASS TD'},
  {k:'carries', t:'RSH'}, {k:'rushYds', t:'RUSH YD'},
  {k:'tgts', t:'TGT'}, {k:'rec', t:'REC'}, {k:'recYds', t:'REC YD'},
  {k:'tdProb', t:'ANY TD', kind:'td'},
];
let state = {pos:'ALL', q:'', sort:'tdProb', game:null};

const el = id => document.getElementById(id);
const fmt = (v, nd=1) => v===null||v===undefined ? '<span class="muted">—</span>'
  : (nd===0 ? Math.round(v) : (Math.round(v*10)/10).toFixed(1));

function renderHead(){
  el('head').innerHTML = COLS.map(c =>
    `<th data-k="${c.k}" class="${state.sort===c.k?'sorted':''}">${c.t}</th>`).join('');
  el('head').querySelectorAll('th').forEach(th => th.onclick = () => {
    state.sort = th.dataset.k; render();
  });
}

function renderGames(){
  el('games').innerHTML = DATA.games.map(g => {
    const c = DATA.colors[g.home] || '#555';
    const on = state.game === g.key ? 'on' : '';
    const wx = g.roof === 'dome' || g.roof === 'closed' ? 'indoors'
      : (g.wind!==null ? `${Math.round(g.wind)} mph wind` : 'outdoors');
    return `<div class="game ${on}" style="--accent:${c}" data-k="${g.key}">
      <div class="matchup"><span>${g.away} @ ${g.home}</span><span class="kick">${g.kick}</span></div>
      <div class="totals">
        <span>${g.away} <b>${g.awayTotal ?? '—'}</b></span>
        <span>${g.home} <b>${g.homeTotal ?? '—'}</b></span>
        <span>O/U <b>${g.total ?? '—'}</b></span>
      </div>
      <div class="totals"><span>${wx}</span></div>
    </div>`;
  }).join('');
  el('games').querySelectorAll('.game').forEach(d => d.onclick = () => {
    state.game = state.game === d.dataset.k ? null : d.dataset.k; render();
  });
  renderDetail();
}

function injBlock(list){
  if(!list || !list.length) return DATA.meta.injury_reports
      ? '<div class="none">No reported injuries</div>'
      : '<div class="none">Injury reports publish Wed–Fri — rerun then</div>';
  const use = p => p.share ? p.share+'% tgt/rush share'
      : (p.snap !== null && p.snap !== undefined ? p.snap+'% snaps' : '');
  return '<ul class="ilist">' + list.map(p => `<li>
      <span class="who"><em class="sd ${p.side}">${p.side}</em>${p.name}<small>${p.pos}${use(p) ? ' · '+use(p) : ''}</small></span>
      <span><span class="inj">${p.injury || ''}</span> <span class="st ${p.status}">${p.status.toUpperCase()}</span></span>
    </li>`).join('') + '</ul>';
}

function watchBlock(list){
  if(!list || !list.length) return '<div class="none">—</div>';
  return '<ul class="wlist">' + list.map(p => `<li>
      <div class="wn">${p.name} <span class="muted">${p.pos}</span></div>
      <div class="wl">${p.line}</div>
      <div class="wy">${p.why}</div>
    </li>`).join('') + '</ul>';
}

function renderDetail(){
  const d = el('detail');
  const g = DATA.games.find(x => x.key === state.game);
  if(!g){ d.className = 'detail'; d.innerHTML = ''; return; }
  const w = g.weather || {};
  const src = (w.sources || []).map(s =>
      `<span class="src">${s.name} ${s.temp ?? '—'}° / ${s.wind ?? '—'}mph</span>`).join('');
  const conf = w.confidence
    ? `<span class="conf ${w.confidence.startsWith('low') ? 'bad' : w.confidence.startsWith('high') ? 'good' : ''}">${w.confidence}</span>`
    : '';
  const wx = `<div class="wx ${w.indoors ? 'dome' : ''}">
      <div>${w.venue ? w.venue + ' — ' : ''}${w.summary || 'Forecast unavailable'} ${conf}</div>
      ${src ? `<div class="srcs">${src}</div>` : ''}
      ${w.impact ? `<span class="imp">${w.impact}</span>` : ''}
    </div>`;
  const col = (side, team) => `<div class="dcol" style="--tcol:${DATA.colors[team]||'#444'}">
      <h4>${team}</h4>
      <div class="wy" style="margin-bottom:6px">Injury report</div>
      ${injBlock(g.injuries[side])}
      <div class="wy" style="margin:14px 0 6px">Key players to watch</div>
      ${watchBlock(g.watch[side])}
    </div>`;
  d.className = 'detail on';
  d.innerHTML = wx + `<div class="dgrid">${col('away', g.away)}${col('home', g.home)}</div>`;
}

function renderControls(){
  const positions = ['ALL','QB','RB','WR','TE'];
  el('posf').innerHTML = positions.map(p =>
    `<button class="pill ${state.pos===p?'on':''}" data-p="${p}">${p}</button>`).join('');
  el('posf').querySelectorAll('.pill').forEach(b => b.onclick = () => {
    state.pos = b.dataset.p; render();
  });
  el('sort').innerHTML = COLS.filter(c=>!['name','mu'].includes(c.k))
    .map(c=>`<option value="${c.k}" ${state.sort===c.k?'selected':''}>sort: ${c.t}</option>`).join('');
  el('sort').onchange = e => { state.sort = e.target.value; render(); };
  el('q').oninput = e => { state.q = e.target.value.toLowerCase(); render(); };
}

function render(){
  renderHead(); renderGames();
  let rows = DATA.players.slice();
  if (state.pos !== 'ALL') rows = rows.filter(r => r.pos === state.pos);
  if (state.game) rows = rows.filter(r => r.game === state.game);
  if (state.q) rows = rows.filter(r => r.name.toLowerCase().includes(state.q));
  rows.sort((a,b) => (b[state.sort] ?? -1) - (a[state.sort] ?? -1));

  const maxTd = Math.max(...DATA.players.map(p => p.tdProb || 0), 0.01);
  el('rows').innerHTML = rows.map(r => {
    const tc = DATA.colors[r.team] || '#444';
    const flags = (r.questionable ? '<span class="q">Q</span>' : '')
      + (r.vacated > 0.08 ? `<span class="vac">+${Math.round(r.vacated*100)}%</span>` : '');
    const w = r.tdProb ? (r.tdProb / maxTd * 100) : 0;
    return `<tr>
      <td class="nm" style="--tc:${tc}">${r.name}${flags}</td>
      <td class="mu">${r.team} vs ${r.opp}</td>
      <td>${fmt(r.passAtt,0)}</td><td>${fmt(r.passYds,0)}</td><td>${fmt(r.passTds)}</td>
      <td>${fmt(r.carries)}</td><td>${fmt(r.rushYds,0)}</td>
      <td>${fmt(r.tgts)}</td><td>${fmt(r.rec)}</td><td>${fmt(r.recYds,0)}</td>
      <td class="tdcell"><div class="bar" style="width:${w}%"></div>
        <span>${r.tdProb!==null ? Math.round(r.tdProb*100)+'%' : '—'}</span></td>
    </tr>`;
  }).join('');
  el('foot').textContent = `${rows.length} players shown · projections are medians, `
    + `not locks · TD probabilities above 50% run hot, shade them down`;
}

function renderScore(){
  const a = DATA.accuracy;
  const box = el('score'), det = el('sdetail');
  if(!a || a.score === null || a.score === undefined){
    box.innerHTML = `<div class="smeta"><div class="sgrade">Model score — not yet graded</div>
      <div class="snote">Run <b>nflproj grade</b> after results land to start the record</div></div>`;
    return;
  }
  const hue = a.score >= 65 ? 'var(--cool)' : a.score >= 45 ? 'var(--warn)' : 'var(--hot)';
  const wk = (a.by_week || []).slice(-12);
  const spark = wk.map(w => `<i class="${w.score < 45 ? 'lo' : ''}" style="height:${Math.max(4, w.score/100*38)}px"
     title="${w.season} wk${w.week}: ${w.score}"></i>`).join('');
  box.innerHTML = `
    <div class="sbig" style="color:${hue}">${a.score}<small>/100</small></div>
    <div class="smeta">
      <div class="sgrade">${a.grade}</div>
      <div class="snote">${a.n.toLocaleString()} graded player-games · ${a.population || ''} · tap for breakdown</div>
    </div>
    <div class="spark">${spark}</div>`;
  box.onclick = () => det.classList.toggle('on');

  const c = a.components || {};
  const bar = (label, o, note) => {
    const pct = o && o.max ? (o.points / o.max * 100) : 0;
    return `<div class="comp">
      <div class="comprow"><span>${label} <span class="muted">${note}</span></span>
        <span>${o ? o.points : '—'} / ${o ? o.max : '—'}</span></div>
      <div class="track"><i style="width:${pct}%"></i></div></div>`;
  };
  const rows = (a.per_stat || []).map(s => `<tr><td>${s.stat.replace('_',' ')}</td>
      <td>${s.n.toLocaleString()}</td><td>${s.mae}</td><td>${s.baseline_mae}</td>
      <td style="color:${s.gain_pct > 0 ? 'var(--cool)' : 'var(--hot)'}">${s.gain_pct}%</td>
      <td>${s.corr ?? '—'}</td></tr>`).join('');
  det.innerHTML =
    bar('Skill vs naive baseline', c.skill_vs_baseline,
        `— ${c.skill_vs_baseline ? c.skill_vs_baseline.mean_gain_pct : 0}% better than a 3-game average`) +
    bar('Touchdown calibration', c.td_calibration,
        `— ${c.td_calibration ? c.td_calibration.mean_abs_error_pts : '—'} pts mean error`) +
    bar('Correlation with outcomes', c.correlation,
        `— mean r=${c.correlation ? c.correlation.mean_corr : '—'}`) +
    `<table class="stat-tb"><thead><tr><th>Stat</th><th>n</th><th>MAE</th>
       <th>Baseline</th><th>Gain</th><th>r</th></tr></thead><tbody>${rows}</tbody></table>
     <div class="snote" style="margin-top:9px">This is a skill score, not "percent correct".
       0 means no better than a trailing 3-game average. 55–70 is what an honest
       NFL projection system looks like; above 85 means something is leaking.</div>`;
}

const m = DATA.meta;
el('title').textContent = m.label || 'Slate projections';
el('meta').innerHTML = `Week <b>${m.week}</b> · <b>${m.games}</b> games · first kickoff <b>${m.first_kickoff||'TBD'}</b>`
  + ` · generated <b>${m.generated}</b>`;
renderScore(); renderControls(); render();

// --- Refresh -------------------------------------------------------------
// Only works when the page is served by `nflproj serve`; a file:// page has
// no way to run Python locally, so the button says so instead of failing.
(function(){
  const btn = el('refresh'), logbox = el('runlog');
  const served = location.protocol === 'http:' || location.protocol === 'https:';
  if(!served){
    btn.classList.add('file');
    btn.textContent = 'Refresh — run `nflproj serve`';
    btn.onclick = () => {
      logbox.classList.add('on');
      logbox.innerHTML = 'This page was opened from disk, so it cannot run anything.\n'
        + 'In Terminal:\n\n  <b>cd ~/nflproj && python3 -m nflproj.cli serve</b>\n\n'
        + 'then use the Refresh button on the page that opens.';
    };
    return;
  }
  let timer = null;
  const setLog = lines => {
    logbox.classList.add('on');
    logbox.textContent = lines.join('\n');
    logbox.scrollTop = logbox.scrollHeight;
  };
  const poll = async () => {
    try {
      const r = await fetch('/api/status', {cache:'no-store'});
      const s = await r.json();
      setLog(s.log && s.log.length ? s.log : ['starting...']);
      if(s.status === 'done'){
        clearInterval(timer);
        btn.textContent = 'Reloading...';
        setTimeout(() => location.reload(), 700);
      } else if(s.status === 'error'){
        clearInterval(timer);
        btn.disabled = false;
        btn.textContent = 'Refresh — retry';
      }
    } catch(e){ /* server restarting mid-run, keep polling */ }
  };
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = 'Refreshing...';
    setLog(['starting...']);
    try {
      await fetch('/api/refresh', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({slate: 'next'})});
      timer = setInterval(poll, 1500);
    } catch(e){
      btn.disabled = false; btn.textContent = 'Refresh — failed to start';
    }
  };
})();

// ---- refresh button -------------------------------------------------
// Only functional when the page is served by `nflproj serve`; a file://
// page cannot run Python, so say so rather than failing silently.
(function(){
  const btn = el('refresh'), stat = el('rstat');
  const served = location.protocol === 'http:' || location.protocol === 'https:';
  if(!served){
    btn.classList.add('file');
    btn.textContent = 'Refresh — needs server';
    btn.title = 'Run: python3 -m nflproj.cli serve';
    btn.onclick = () => { stat.textContent = 'run: python3 -m nflproj.cli serve'; };
    return;
  }
  let timer = null;
  const poll = () => fetch('/status').then(r => r.json()).then(s => {
    if(s.running){
      stat.className = '';
      stat.textContent = s.step || 'working...';
      btn.disabled = true;
      btn.textContent = 'Refreshing';
    } else {
      clearInterval(timer); timer = null;
      btn.disabled = false; btn.textContent = 'Refresh';
      if(s.error){ stat.className = 'err'; stat.textContent = s.error; }
      else if(s.finished){ stat.textContent = 'updated — reloading'; setTimeout(() => location.reload(), 600); }
    }
  }).catch(() => { clearInterval(timer); timer = null;
    btn.disabled = false; btn.textContent = 'Refresh';
    stat.className = 'err'; stat.textContent = 'server not responding'; });

  btn.onclick = () => {
    btn.disabled = true; btn.textContent = 'Refreshing';
    stat.className = ''; stat.textContent = 'starting...';
    fetch('/run', {method:'POST'}).then(r => {
      if(r.status === 409){ stat.textContent = 'already running'; }
      if(!timer) timer = setInterval(poll, 1200);
    }).catch(() => { btn.disabled = false; btn.textContent = 'Refresh';
      stat.className = 'err'; stat.textContent = 'could not reach server'; });
  };
  // if a run is already in flight when the page loads, attach to it
  fetch('/status').then(r => r.json()).then(s => { if(s.running && !timer) timer = setInterval(poll, 1200); });
})();
</script>
</body>
</html>
"""
