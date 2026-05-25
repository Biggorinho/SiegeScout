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
            # Lines without a timestamp prefix (e.g. the un-timestamped
            # session-marker banner) are skipped — session boundaries are
            # already handled by split_sessions() before we get here.
            continue
        ts = m.group("ts")
        rest = m.group("rest")
        if not session_start_ts:
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

        # Vote-reset success: every successful reset ends with the bot
        # clicking a node on the world map, which prints exactly once per
        # successful attempt. The older "Vote reset triggered" form is kept
        # for backward-compat with logs from earlier bot versions.
        if "[RST] satanic node picked at" in rest or "Vote reset triggered" in rest:
            vote_resets.append({"timestamp": ts})
            continue

        # Each failed retry prints two lines ("...failed after all retries"
        # and "...vote reset failed xN -> cooldown ..."). Count only the
        # first to avoid double-counting one event.
        if "[RST] vote reset failed after all retries" in rest:
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


import math
import os


def aggregate_counters(zones: list[dict]) -> dict:
    """Return {counter_name: {total, mean, max, count_zones, values}}."""
    if not zones:
        return {}
    by_name: dict[str, list[int]] = {}
    for z in zones:
        for k, v in z.get("counters", {}).items():
            by_name.setdefault(k, []).append(int(v))
    result = {}
    n = len(zones)
    for name, values in by_name.items():
        # Pad missing zones with zeros so the mean reflects the full session.
        padded = values + [0] * (n - len(values))
        total = sum(padded)
        result[name] = {
            "total": total,
            "mean": total / n,
            "max": max(padded),
            "count_zones": n,
            "values": padded,
        }
    return result


def find_outliers(zones: list[dict], counter: str) -> list[tuple[int, int, str]]:
    """Return [(zone_index, value, zone_name), ...] for zones where the named
    counter is > max(mean + 1.5*stdev, 5). Sorted by value descending."""
    n = len(zones)
    if n == 0:
        return []
    values = [int(z.get("counters", {}).get(counter, 0)) for z in zones]
    mean = sum(values) / n
    # Population std-dev (n divisor, not n-1 - fewer zones, stable enough).
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    threshold = max(mean + 1.5 * std, 5)
    out = []
    for i, v in enumerate(values):
        if v > threshold:
            out.append((i, v, zones[i].get("name", f"zone {i}")))
    out.sort(key=lambda t: -t[1])
    return out


# Snapshot filename pattern: <event>_<unix_ms>.png
# Where <event> may contain underscores (e.g. "qte_miss_fallback") so we
# split on the LAST underscore that precedes the digits-only ms timestamp.
_SNAP_RE = re.compile(r"^(?P<event>.+)_(?P<ms>\d+)\.png$")


def list_snapshots(debug_dir: str) -> list[dict]:
    """Return [{event, unix_ms, path}, ...] sorted by unix_ms ascending."""
    if not os.path.isdir(debug_dir):
        return []
    out = []
    for name in os.listdir(debug_dir):
        m = _SNAP_RE.match(name)
        if not m:
            continue
        out.append({
            "event": m.group("event"),
            "unix_ms": int(m.group("ms")),
            "path": os.path.join(debug_dir, name).replace("\\", "/"),
        })
    out.sort(key=lambda d: d["unix_ms"])
    return out


def map_snapshots_to_zones(zones: list[dict], snapshots: list[dict]) -> dict:
    """Return {zone_index: [snapshot_dict, ...]} based on unix_ms in
    [zone.start_ms, zone.end_ms]. Zones without a start_ms/end_ms are
    skipped (the caller is responsible for converting ISO ts -> unix_ms)."""
    out: dict[int, list[dict]] = {i: [] for i in range(len(zones))}
    for snap in snapshots:
        for i, z in enumerate(zones):
            s = z.get("start_ms")
            e = z.get("end_ms")
            if s is None or e is None:
                continue
            if s <= snap["unix_ms"] <= e:
                out[i].append(snap)
                break
    return out


def _duration_human(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _iso_seconds_between(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    import datetime
    try:
        da = datetime.datetime.fromisoformat(a)
        db = datetime.datetime.fromisoformat(b)
        return (db - da).total_seconds()
    except ValueError:
        return 0.0


def render_report(events: dict, snapshots: list[dict],
                  zone_snapshots: dict[int, list[dict]]) -> str:
    """Return the full Markdown report as a single string."""
    zones = events.get("zones", [])
    deaths = events.get("deaths", [])
    stucks = events.get("stucks", [])
    vote_resets = events.get("vote_resets", [])
    vote_reset_failures = events.get("vote_reset_failures", [])
    start_ts = events.get("session_start_ts")
    end_ts = events.get("session_end_ts")
    runtime = _duration_human(_iso_seconds_between(start_ts, end_ts))

    aggs = aggregate_counters(zones)
    lines: list[str] = []
    lines.append(f"# SiegeScout Session Report — {start_ts} → {end_ts}\n")
    lines.append(
        f"**Runtime:** {runtime}  |  **Zones:** {len(zones)}  |  "
        f"**Deaths:** {len(deaths)}  |  "
        f"**Vote-resets:** {len(vote_resets)} ok / {len(vote_reset_failures)} failed\n"
    )

    # Counter summary table.
    lines.append("\n## Counter Summary\n")
    if not aggs:
        lines.append("(none)\n")
    else:
        lines.append("| Counter | Total | Mean/zone | Max | Outlier zones |")
        lines.append("|---|---|---|---|---|")
        for name in sorted(aggs.keys()):
            a = aggs[name]
            outs = find_outliers(zones, counter=name)
            out_str = ", ".join(f"{zname}({val})" for _, val, zname in outs[:3]) if outs else "—"
            lines.append(
                f"| {name} | {a['total']} | {a['mean']:.1f} | {a['max']} | {out_str} |"
            )

    # Problem zones (statistical outliers only).
    lines.append("\n## Problem Zones\n")
    problem_indices = set()
    for name in aggs.keys():
        for idx, _val, _zname in find_outliers(zones, counter=name):
            problem_indices.add(idx)
    if not problem_indices:
        lines.append("(none)\n")
    else:
        for idx in sorted(problem_indices):
            z = zones[idx]
            lines.append(
                f"### Zone {idx + 1} — {z.get('name', '?')} "
                f"({_duration_human(z.get('duration_seconds', 0))})"
            )
            counter_dump = " ".join(
                f"{k}={v}" for k, v in sorted(z.get("counters", {}).items())
            )
            lines.append(f"- Counters: {counter_dump}")
            snaps = zone_snapshots.get(idx, [])
            if snaps:
                lines.append("- Snapshots:")
                for s in snaps[:3]:
                    lines.append(f"  - {s['path']}")
            else:
                lines.append("- Snapshots: (none)")
            lines.append("")

    # Stuck patterns.
    lines.append("\n## Stuck Patterns\n")
    if not stucks:
        lines.append("(none)\n")
    else:
        by_angle: dict[int, int] = {}
        for s in stucks:
            by_angle[s["octant_degrees"]] = by_angle.get(s["octant_degrees"], 0) + 1
        ordered = sorted(by_angle.items(), key=lambda t: -t[1])
        lines.append("- Octant-escape angles: " +
                     ", ".join(f"{a}° ({c}×)" for a, c in ordered))

    # Death events.
    lines.append("\n## Death Events\n")
    if not deaths:
        lines.append("(none)\n")
    else:
        for i, d in enumerate(deaths, 1):
            lines.append(f"{i}. {d['timestamp']} — {d['line']}")

    # Vote-reset reliability.
    lines.append("\n## Vote-Reset Reliability\n")
    total = len(vote_resets) + len(vote_reset_failures)
    if total == 0:
        lines.append("(none)\n")
    else:
        ok = len(vote_resets)
        pct = (ok * 100) // total
        lines.append(f"- {total} attempted, {ok} succeeded ({pct}% success).")
        if vote_reset_failures:
            lines.append(f"- Failures: {len(vote_reset_failures)} (see log timestamps)")

    return "\n".join(lines) + "\n"


def _iso_to_unix_ms(ts: str | None) -> int | None:
    if not ts:
        return None
    import datetime
    try:
        return int(datetime.datetime.fromisoformat(ts).timestamp() * 1000)
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Analyze a SiegeScout debug session.")
    ap.add_argument("--log", default="debug/bot.log",
                    help="Path to bot.log (default: debug/bot.log)")
    ap.add_argument("--debug-dir", default="debug",
                    help="Directory with *.png snapshots (default: debug)")
    ap.add_argument("--out", default="debug/SESSION_REPORT.md",
                    help="Output path (default: debug/SESSION_REPORT.md)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.log):
        print(f"[ERROR] {args.log} not found. Run the bot first — gui.py "
              f"writes bot.log on every session.")
        return 1

    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    session_lines = split_sessions(lines)
    events = parse_log_lines(session_lines)

    # Convert each zone's ISO start_ts/end_ts to unix_ms so snapshots can be
    # mapped purely on the integer timeline.
    for z in events["zones"]:
        z["start_ms"] = _iso_to_unix_ms(z.get("start_ts"))
        z["end_ms"] = _iso_to_unix_ms(z.get("end_ts"))

    snapshots = list_snapshots(args.debug_dir)
    zone_snapshots = map_snapshots_to_zones(events["zones"], snapshots)

    report = render_report(events, snapshots, zone_snapshots)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Wrote {args.out} ({len(report)} bytes, {len(events['zones'])} zones, "
          f"{len(snapshots)} snapshots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
