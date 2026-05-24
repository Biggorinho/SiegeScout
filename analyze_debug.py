"""SiegeScout debug-session analyzer.

Reads debug/bot.log + debug/*.png snapshots from a single bot session
and emits debug/SESSION_REPORT.md with counter aggregates, problem-zone
outliers, death analysis, stuck patterns, vote-reset reliability, and
resource-targeting effectiveness.

Run: python analyze_debug.py
     python analyze_debug.py --log custom.log --out my_report.md
"""

from __future__ import annotations

import re

# Lines start with an ISO-8601 timestamp (e.g. "2026-05-24T11:09:22.345  ").
_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s+(?P<rest>.*)$")

_CNT_RE = re.compile(
    r"\[CNT\]\s+zone=(?P<zone>[\d.]+)s\s+(?P<kvs>(?:\w+=\d+\s*)+)"
)

_KV_RE = re.compile(r"(\w+)=(\d+)")

_SESSION_MARKER = "===== Bot session started"


def parse_cnt_line(line: str) -> dict | None:
    """Return {timestamp, zone_seconds, counters: {name: int, ...}} or None."""
    if not line:
        return None
    m = _TS_RE.match(line)
    if not m:
        return None
    ts = m.group("ts")
    rest = m.group("rest")
    cnt = _CNT_RE.search(rest)
    if not cnt:
        return None
    zone_seconds = float(cnt.group("zone"))
    counters = {k: int(v) for k, v in _KV_RE.findall(cnt.group("kvs"))}
    return {"timestamp": ts, "zone_seconds": zone_seconds, "counters": counters}


def split_sessions(lines: list[str]) -> list[str]:
    """Return ONLY the lines belonging to the last session (marked by
    `===== Bot session started`). If no marker exists, return all lines."""
    last_start = -1
    for i, ln in enumerate(lines):
        if _SESSION_MARKER in ln:
            last_start = i
    if last_start < 0:
        return lines
    return lines[last_start:]


def parse_log_lines(lines: list[str]) -> dict:
    """Parse a list of log lines (one session) into structured events.

    Returns a dict with keys: zones, deaths, stucks, vote_resets,
    vote_reset_failures, hp_events, session_start_ts, session_end_ts.
    """
    zones: list[dict] = []
    deaths: list[dict] = []
    stucks: list[dict] = []
    vote_resets: list[dict] = []
    vote_reset_failures: list[dict] = []
    hp_events: list[dict] = []
    session_start_ts: str | None = None
    session_end_ts: str | None = None

    pending_zone_name: str | None = None
    pending_zone_start_ts: str | None = None

    for raw in lines:
        m = _TS_RE.match(raw)
        if not m:
            # Lines without a timestamp prefix (e.g. the session-marker banner)
            # are still useful for session boundaries.
            if _SESSION_MARKER in raw and session_start_ts is None:
                session_start_ts = ""
            continue
        ts = m.group("ts")
        rest = m.group("rest")
        if session_start_ts is None:
            session_start_ts = ts
        session_end_ts = ts

        if "Zone entered:" in rest:
            pending_zone_name = rest.split("Zone entered:", 1)[1].strip()
            pending_zone_start_ts = ts
            continue

        cnt = parse_cnt_line(raw)
        if cnt is not None:
            zones.append({
                "name": pending_zone_name or "(unknown)",
                "start_ts": pending_zone_start_ts or ts,
                "end_ts": ts,
                "duration_seconds": cnt["zone_seconds"],
                "counters": cnt["counters"],
            })
            pending_zone_name = None
            pending_zone_start_ts = None
            continue

        if "[STUCK] octant-escape was=" in rest:
            angle_m = re.search(r"was=(-?\d+)°", rest)
            if angle_m:
                stucks.append({"timestamp": ts, "octant_degrees": int(angle_m.group(1))})
            continue

        if "[DEATH]" in rest:
            deaths.append({"timestamp": ts, "line": rest})
            continue

        if "Vote reset triggered" in rest:
            vote_resets.append({"timestamp": ts})
            continue

        if "[RST] vote reset failed" in rest:
            vote_reset_failures.append({"timestamp": ts, "line": rest})
            continue

        if rest.startswith("[HP]") or rest.startswith("HP "):
            hp_events.append({"timestamp": ts, "line": rest})
            continue

    return {
        "zones": zones,
        "deaths": deaths,
        "stucks": stucks,
        "vote_resets": vote_resets,
        "vote_reset_failures": vote_reset_failures,
        "hp_events": hp_events,
        "session_start_ts": session_start_ts,
        "session_end_ts": session_end_ts,
    }
