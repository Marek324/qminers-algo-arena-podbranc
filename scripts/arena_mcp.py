"""Read-only MCP server for the Algo Trade Arena site (arena.qminers.com): leaderboard, rounds and our submissions.

Run by Claude Code through .mcp.json (`uv run --with mcp python scripts/arena_mcp.py`); the `mcp` package is pulled in
only for this process, the project's dependencies are unchanged.

Authentication: the site logs in by magic link and keeps a bearer token in the browser. Copy it once:
  1. open arena.qminers.com, logged in;
  2. DevTools console (F12): localStorage.getItem('qminers.accessToken')
  3. export ARENA_TOKEN='<that value>' in the shell that starts Claude Code (never commit it).

Read-only by construction: the only request this file can make is a GET to one of ALLOWED paths. Submitting a
strategy or marking a submission final stays a manual action on the site.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import urllib.error
import urllib.request

from mcp.server.mcpserver import MCPServer

BASE = 'https://arena.qminers.com/api'
ALLOWED = (
    re.compile(r'^/auth/me$'),
    re.compile(r'^/rounds/$'),
    re.compile(r'^/rounds/current$'),
    re.compile(r'^/rounds/[A-Za-z0-9_-]+/leaderboard$'),
    re.compile(r'^/evaluations/$'),
    re.compile(r'^/evaluations/[A-Za-z0-9_-]+$'),
)
MAX_CHARS = 30_000

mcp = MCPServer("arena")


def _get(path: str):
    """GET an allowed API path with the bearer token; returns parsed JSON (or text)."""
    if not any(p.match(path) for p in ALLOWED):
        raise ValueError(f'path not allowed (read-only allow-list): {path}')
    token = re.sub(r'\s+', '', os.environ.get('ARENA_TOKEN', ''))  # a wrapped paste can leave a newline inside
    if not token:
        raise RuntimeError("ARENA_TOKEN is not set. In the logged-in site's DevTools console run "
                           "localStorage.getItem('qminers.accessToken') and export ARENA_TOKEN='<value>' before starting Claude Code.")
    req = urllib.request.Request(BASE + path, method='GET', headers={'Authorization': f'Bearer {token}', 'Accept': 'application/json',
                                                                     'User-Agent': 'arena-readonly-mcp'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace')[:300]
        if e.code == 401:
            raise RuntimeError(f'401 from the arena: the token is missing or expired; copy a fresh one ({detail})') from None
        raise RuntimeError(f'HTTP {e.code} from {path}: {detail}') from None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _strip_series(obj):
    """Drop long time series (pnlSeries etc.) but keep their length and last value."""
    if isinstance(obj, list):
        return [_strip_series(v) for v in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, list) and len(v) > 20 and all(isinstance(x, dict) for x in v[:3]):
                out[k] = f'<{len(v)} points, last {v[-1]}>'
            else:
                out[k] = _strip_series(v)
        return out
    return obj


def _latest_round_id() -> str:
    """The active round, or else the most recent round with a published leaderboard."""
    try:
        cur = _get('/rounds/current')
        if isinstance(cur, dict) and cur.get('id'):
            return str(cur['id'])
    except RuntimeError:
        pass
    rounds = [r for r in _rows(_get('/rounds/')) if r.get('isLeaderboardPublished', True)]
    rounds.sort(key=lambda r: str(r.get('startAt', '')))
    return str(rounds[-1]['id']) if rounds else ''


def _round_id(name_or_id: str) -> str:
    """Accept a round id or a round name such as 'Round 1' / 'round1' / '1'."""
    if not name_or_id:
        return _latest_round_id()
    key = name_or_id.lower().replace(' ', '')
    for r in _rows(_get('/rounds/')):
        name = str(r.get('name', '')).lower().replace(' ', '')
        if key in (str(r.get('id', '')).lower(), name, name.replace('round', '')):
            return str(r['id'])
    return name_or_id


def _dump(obj) -> str:
    text = json.dumps(_strip_series(obj), indent=1, ensure_ascii=False) if not isinstance(obj, str) else obj
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + f'\n... (truncated, {len(text):,} chars)'


def _rows(obj) -> list[dict]:
    """The list of entries in a response, whether it is a list or wrapped in an object."""
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _first(row: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    low = {k.lower(): v for k, v in row.items()}
    for k in keys:
        if k.lower() in low and low[k.lower()] is not None:
            return low[k.lower()]
    return None


PNL_KEYS = ('pnl', 'publicPnl', 'public_pnl', 'bestPnl', 'best_pnl', 'score', 'value')
TEAM_KEYS = ('teamName', 'team_name', 'team', 'name', 'displayName')


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return _num(_first(v, PNL_KEYS))
    try:
        return float(str(v).replace(',', '').replace('+', ''))
    except (TypeError, ValueError):
        return None


@mcp.tool()
def arena_me() -> str:
    """Who we are logged in as (account and team), from GET /auth/me."""
    return _dump(_get('/auth/me'))


@mcp.tool()
def arena_rounds() -> str:
    """All rounds (ids, names, dates, status), from GET /rounds/."""
    return _dump(_get('/rounds/'))


@mcp.tool()
def arena_current_round() -> str:
    """The current round, from GET /rounds/current (says so when no round is active)."""
    try:
        return _dump(_get('/rounds/current'))
    except RuntimeError as e:
        if '404' in str(e):
            return 'No round is currently active. arena_rounds lists all rounds.'
        raise


@mcp.tool()
def arena_leaderboard(round_name: str = '', top: int = 20) -> str:
    """Leaderboard of a round with each team's z-score as the final score computes it: PnL standardized against the
    teams with PnL > 0 (mean and sample sd, n-1; teams at or below 0 count as PnL 0). round_name: 'Round 1', '1', a round id, or empty for the
    active round (else the latest published one)."""
    round_id = _round_id(round_name)
    if not round_id:
        return 'No rounds found.'
    try:
        raw = _get(f'/rounds/{round_id}/leaderboard')
    except RuntimeError as e:
        if round_name or 'HTTP 403' not in str(e):
            raise
        # the active round's leaderboard is hidden until it is published: show the latest published round instead
        published = [r for r in _rows(_get('/rounds/')) if r.get('isLeaderboardPublished') and str(r.get('id')) != round_id]
        published.sort(key=lambda r: str(r.get('startAt', '')))
        if not published:
            raise
        round_id = str(published[-1]['id'])
        raw = _get(f'/rounds/{round_id}/leaderboard')
    rows = _rows(raw)
    parsed = [(str(_first(r, TEAM_KEYS)), _num(_first(r, PNL_KEYS))) for r in rows]
    parsed = [(t, p) for t, p in parsed if p is not None]
    if not parsed:
        return 'Leaderboard format not recognised; raw response:\n' + _dump(raw)
    parsed.sort(key=lambda x: -x[1])
    pos = [p for _, p in parsed if p > 0]
    mean = statistics.fmean(pos) if pos else 0.0
    sd = statistics.stdev(pos) if len(pos) > 1 else 0.0  # sample sd (n-1): matches the official board
    lines = [f'round {round_id}: {len(parsed)} teams, {len(pos)} with PnL > 0, mean {mean:,.0f}, sd {sd:,.0f}', '',
             f'{"#":>3}  {"team":<40}{"PnL":>14}{"z":>8}']
    for i, (team, p) in enumerate(parsed[: max(top, 1)], 1):
        z = ((max(p, 0.0) - mean) / sd) if sd else 0.0
        lines.append(f'{i:>3}  {team[:40]:<40}{p:>14,.2f}{z:>+8.2f}')
    return '\n'.join(lines)


@mcp.tool()
def arena_submissions(limit: int = 20) -> str:
    """Our submissions (newest first as the site returns them), from GET /evaluations/."""
    raw = _get('/evaluations/')
    rows = _rows(raw)
    return _dump(rows[: max(limit, 1)] if rows else raw)


@mcp.tool()
def arena_submission(evaluation_id: str) -> str:
    """One submission's details and metrics (PnL, penalty, fills, inventory quantiles), from GET /evaluations/{id}."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', evaluation_id or ''):
        return 'evaluation_id must be the id shown on the site (letters, digits, - or _)'
    return _dump(_get(f'/evaluations/{evaluation_id}'))


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
STORAGE_HOST = re.compile(r'^https://[a-z0-9.-]+\.r2\.cloudflarestorage\.com/')


def _fetch_file(url: str) -> bytes:
    """GET a download link the API returned (presigned storage URL only)."""
    if not STORAGE_HOST.match(url or ''):
        raise ValueError(f'not an arena storage link: {url[:60]}')
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'arena-readonly-mcp'}), timeout=60) as r:
        return r.read()


def _save_evaluation(ev: dict, out_root: str) -> str:
    """Save one submission: strategy.py plus the unpacked results zip (result.json, pnl.csv, metadata.json)."""
    import io
    import zipfile

    d = _get(f'/evaluations/{ev["id"]}')
    folder = os.path.join(out_root, f'{str(ev.get("createdAt", ""))[11:16].replace(":", "")}_{str(ev["id"])[:8]}')
    os.makedirs(folder, exist_ok=True)
    got = []
    if d.get('strategyFile'):
        with open(os.path.join(folder, 'strategy.py'), 'wb') as f:
            f.write(_fetch_file(d['strategyFile']))
        got.append('strategy.py')
    if d.get('resultFileZip'):
        z = zipfile.ZipFile(io.BytesIO(_fetch_file(d['resultFileZip'])))
        for name in z.namelist():
            base = os.path.basename(name)
            if base and not base.startswith('.') and '..' not in name:
                with open(os.path.join(folder, base), 'wb') as f:
                    f.write(z.read(name))
                got.append(base)
    meta = {k: d.get(k) for k in ('id', 'createdAt', 'type', 'status', 'score', 'isFinal', 'sourceEvaluationId', 'metrics')}
    meta['createdBy'] = (d.get('createdBy') or {}).get('name')
    with open(os.path.join(folder, 'evaluation.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    who = meta['createdBy'] or '?'
    return f'{folder}: score {d.get("score")}, {d.get("type")}, by {who}, saved {", ".join(got) or "nothing (still running?)"}'


@mcp.tool()
def arena_download(round_name: str = '', evaluation_id: str = '') -> str:
    """Download submissions for local analysis: each one's strategy.py, the arena's result.json / pnl.csv /
    metadata.json from its results zip, and evaluation.json (score, type, author, metrics), into
    data/validation_<round>/<HHMM>_<id8>/ (the anchor layout scripts/twin_check.py reads). Give evaluation_id for one
    submission, or round_name ('Round 3', '3', empty for the active round) for all of that round's submissions.
    Read-only: GETs to the API and to the download links it returns; writes only under data/."""
    if evaluation_id:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', evaluation_id):
            return 'evaluation_id must be the id shown on the site (letters, digits, - or _)'
        ev = _get(f'/evaluations/{evaluation_id}')
        rows = [ev]
        rid = str(ev.get('roundId', ''))
    else:
        rid = _round_id(round_name)
        rows = [r for r in _rows(_get('/evaluations/')) if str(r.get('roundId')) == rid]
    names = {str(r.get('id')): str(r.get('name', '')) for r in _rows(_get('/rounds/'))}
    tag = re.sub(r'[^a-z0-9]+', '', names.get(rid, rid[:8]).lower()) or rid[:8]
    out_root = os.path.join(DATA_DIR, f'validation_{tag}')
    if not rows:
        return f'no submissions found for round {names.get(rid, rid)}'
    lines = []
    for ev in rows:
        if str(ev.get('status')) != 'COMPLETED':
            lines.append(f'{ev.get("id")}: {ev.get("status")}, skipped (not finished)')
            continue
        lines.append(_save_evaluation(ev, out_root))
    return '\n'.join(lines)


if __name__ == '__main__':
    mcp.run()
