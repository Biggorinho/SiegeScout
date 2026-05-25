import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_debug import parse_cnt_line, split_sessions, parse_log_lines
from analyze_debug import (
    aggregate_counters, find_outliers,
    list_snapshots, map_snapshots_to_zones,
)


def test_parse_cnt_line_returns_dict_with_all_six_counters():
    line = ("2026-05-24T11:09:22.345  [CNT] zone=3.8s "
            "attack_no_healthbar_after_2s=0 enemy_dot_lost_before_confirm=15 "
            "qte_press_missed_window=0 resource_target_disappeared_early=1 "
            "resource_target_timeout=0 item_pickup_fired_no_target=2")
    result = parse_cnt_line(line)
    assert result is not None
    assert result["zone_seconds"] == 3.8
    assert result["timestamp"] == "2026-05-24T11:09:22.345"
    assert result["counters"]["enemy_dot_lost_before_confirm"] == 15
    assert result["counters"]["item_pickup_fired_no_target"] == 2
    assert result["counters"]["qte_press_missed_window"] == 0


def test_parse_cnt_line_returns_none_for_non_cnt_line():
    assert parse_cnt_line("2026-05-24T11:09:22.345  [DETECT] 5 enemies") is None
    assert parse_cnt_line("garbage line") is None
    assert parse_cnt_line("") is None


def test_parse_cnt_line_handles_unknown_counter_keys():
    line = ("2026-05-24T11:09:22.345  [CNT] zone=1.0s "
            "enemy_dot_lost_before_confirm=3 future_counter=99")
    result = parse_cnt_line(line)
    assert result is not None
    assert result["counters"]["enemy_dot_lost_before_confirm"] == 3
    assert result["counters"]["future_counter"] == 99   # passed through


def test_split_sessions_returns_only_last_session_by_default():
    lines = [
        "2026-05-24T10:00:00.000  ===== Bot session started ... =====",
        "2026-05-24T10:00:01.000  [STARTUP] ...",
        "2026-05-24T10:05:00.000  ===== Bot session started ... =====",
        "2026-05-24T10:05:01.000  [STARTUP] second session",
        "2026-05-24T10:05:02.000  Bot stopped",
    ]
    result = split_sessions(lines)
    assert len(result) == 3   # last session has 3 lines
    assert "second session" in result[1]


def test_split_sessions_handles_log_without_session_marker():
    lines = ["random line 1", "random line 2"]
    result = split_sessions(lines)
    assert result == lines


def test_parse_log_lines_extracts_zone_summaries():
    lines = [
        "2026-05-24T10:00:00.000  ===== Bot session started =====",
        "2026-05-24T10:00:01.000  Zone entered: Act 6 : Highland Graveyard",
        "2026-05-24T10:00:30.000  [CNT] zone=29.0s "
        "attack_no_healthbar_after_2s=0 enemy_dot_lost_before_confirm=5 "
        "qte_press_missed_window=0 resource_target_disappeared_early=0 "
        "resource_target_timeout=0 item_pickup_fired_no_target=0",
        "2026-05-24T10:00:31.000  Vote reset triggered",
    ]
    events = parse_log_lines(lines)
    assert len(events["zones"]) == 1
    z = events["zones"][0]
    assert z["name"] == "Act 6 : Highland Graveyard"
    assert z["counters"]["enemy_dot_lost_before_confirm"] == 5
    assert z["duration_seconds"] == 29.0


def test_parse_log_lines_extracts_deaths_and_stucks_and_resets():
    lines = [
        "2026-05-24T10:00:00.000  ===== Bot session started =====",
        "2026-05-24T10:00:10.000  [STUCK] octant-escape was=115°",
        "2026-05-24T10:00:20.000  [DEATH] Respawn + Vote Reset",
        "2026-05-24T10:00:25.000  Vote reset triggered",
        "2026-05-24T10:00:30.000  [RST] vote reset failed x1 -> cooldown 8s",
    ]
    events = parse_log_lines(lines)
    assert len(events["stucks"]) == 1
    assert events["stucks"][0]["octant_degrees"] == 115
    assert len(events["deaths"]) == 1
    assert len(events["vote_resets"]) == 1
    assert len(events["vote_reset_failures"]) == 1


def test_aggregate_counters_empty_zones_returns_zeros():
    result = aggregate_counters([])
    assert result == {}


def test_aggregate_counters_sums_means_max():
    zones = [
        {"counters": {"a": 10, "b": 0}},
        {"counters": {"a": 20, "b": 5}},
        {"counters": {"a": 30, "b": 0}},
    ]
    result = aggregate_counters(zones)
    assert result["a"]["total"] == 60
    assert result["a"]["mean"] == 20.0
    assert result["a"]["max"] == 30
    assert result["a"]["count_zones"] == 3
    assert result["b"]["total"] == 5
    assert result["b"]["max"] == 5


def test_find_outliers_flags_zones_above_mean_plus_1_5_sigma_and_5():
    # Counter "x" values across zones: 0, 0, 0, 0, 50 -> mean=10, std~=20
    # threshold = max(mean + 1.5*std, 5) -> ~40
    # 50 > 40 -> outlier (zone index 4).
    zones = [
        {"name": f"z{i}", "counters": {"x": v}}
        for i, v in enumerate([0, 0, 0, 0, 50])
    ]
    outliers = find_outliers(zones, counter="x")
    assert len(outliers) == 1
    assert outliers[0][0] == 4   # zone index
    assert outliers[0][1] == 50


def test_find_outliers_skips_small_counter_values_under_5():
    # If everything is 0-3, no outlier even if one is statistically high.
    zones = [{"counters": {"x": v}} for v in [0, 0, 0, 0, 3]]
    outliers = find_outliers(zones, counter="x")
    assert outliers == []


def test_list_snapshots_reads_filenames_with_event_and_ms_timestamp(tmp_path):
    (tmp_path / "enter_attack_1779568283961.png").write_bytes(b"")
    (tmp_path / "periodic_1779568289663.png").write_bytes(b"")
    (tmp_path / "qte_miss_fallback_1779568300000.png").write_bytes(b"")
    (tmp_path / "ignore_me.txt").write_text("not a png")
    snaps = list_snapshots(str(tmp_path))
    assert len(snaps) == 3
    # Sorted by timestamp ascending.
    assert snaps[0]["event"] == "enter_attack"
    assert snaps[0]["unix_ms"] == 1779568283961
    # "qte_miss_fallback" - event name is everything before the trailing _<ms>.
    assert snaps[2]["event"] == "qte_miss_fallback"


def test_map_snapshots_to_zones_assigns_each_snapshot_to_zone_by_timespan():
    # Two zones: [1000, 2000] and [2500, 3500] (in unix ms).
    zones = [
        {"name": "z1", "start_ms": 1000, "end_ms": 2000},
        {"name": "z2", "start_ms": 2500, "end_ms": 3500},
    ]
    snaps = [
        {"event": "a", "unix_ms": 1500, "path": "a.png"},   # in z1
        {"event": "b", "unix_ms": 2700, "path": "b.png"},   # in z2
        {"event": "c", "unix_ms": 5000, "path": "c.png"},   # outside both
    ]
    mapping = map_snapshots_to_zones(zones, snaps)
    assert len(mapping[0]) == 1 and mapping[0][0]["path"] == "a.png"
    assert len(mapping[1]) == 1 and mapping[1][0]["path"] == "b.png"


from analyze_debug import render_report


def test_render_report_minimal_no_data():
    sections = render_report({
        "session_start_ts": "2026-05-24T10:00:00.000",
        "session_end_ts":   "2026-05-24T10:00:10.000",
        "zones": [],
        "deaths": [],
        "stucks": [],
        "vote_resets": [],
        "vote_reset_failures": [],
        "hp_events": [],
    }, snapshots=[], zone_snapshots={})
    assert "# SiegeScout Session Report" in sections
    assert "**Zones:** 0" in sections
    assert "**Deaths:** 0" in sections
    assert "(none)" in sections   # at least one empty-section placeholder


def test_render_report_with_zones_lists_counter_table_and_problem_zones():
    zones = [
        {"name": "z1", "duration_seconds": 30.0,
         "counters": {"enemy_dot_lost_before_confirm": 5},
         "start_ms": 1000, "end_ms": 31000},
        {"name": "z2", "duration_seconds": 25.0,
         "counters": {"enemy_dot_lost_before_confirm": 50},
         "start_ms": 31000, "end_ms": 56000},
        {"name": "z3", "duration_seconds": 20.0,
         "counters": {"enemy_dot_lost_before_confirm": 0},
         "start_ms": 56000, "end_ms": 76000},
    ]
    sections = render_report({
        "session_start_ts": "2026-05-24T10:00:00.000",
        "session_end_ts":   "2026-05-24T10:01:16.000",
        "zones": zones,
        "deaths": [],
        "stucks": [],
        "vote_resets": [{"timestamp": "2026-05-24T10:00:30.000"}],
        "vote_reset_failures": [],
        "hp_events": [],
    }, snapshots=[], zone_snapshots={1: [{"event": "periodic", "path": "debug/p_1.png"}]})
    assert "enemy_dot_lost_before_confirm" in sections
    assert "z2" in sections
    assert "## Problem Zones" in sections
    assert "debug/p_1.png" in sections
