import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_debug import parse_cnt_line, split_sessions, parse_log_lines


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
