import io
import os
import sys
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state._detection_counters import DetectionCounters


def test_new_counters_all_zero():
    c = DetectionCounters()
    snap = c.snapshot()
    assert snap == {
        "attack_no_healthbar_after_2s": 0,
        "enemy_dot_lost_before_confirm": 0,
        "qte_press_missed_window": 0,
        "resource_target_disappeared_early": 0,
        "resource_target_timeout": 0,
        "item_pickup_fired_no_target": 0,
    }


def test_increment_and_snapshot():
    c = DetectionCounters()
    c.attack_no_healthbar_after_2s = 2
    c.qte_press_missed_window = 7
    snap = c.snapshot()
    assert snap["attack_no_healthbar_after_2s"] == 2
    assert snap["qte_press_missed_window"] == 7


def test_reset_zeroes_and_updates_zone_start():
    c = DetectionCounters(zone_started_at=100.0)
    c.attack_no_healthbar_after_2s = 5
    c.reset(now=200.0)
    assert c.zone_started_at == 200.0
    snap = c.snapshot()
    assert all(v == 0 for v in snap.values())


def test_log_dump_format():
    c = DetectionCounters(zone_started_at=0.0)
    c.attack_no_healthbar_after_2s = 1
    c.qte_press_missed_window = 3
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c.log_dump(now=12.5)
    out = buf.getvalue().strip()
    assert out.startswith("[CNT] zone=12.5s")
    assert "attack_no_healthbar_after_2s=1" in out
    assert "qte_press_missed_window=3" in out


def test_snapshot_keys_match_field_names():
    c = DetectionCounters()
    snap_keys = set(c.snapshot().keys())
    fields = {
        "attack_no_healthbar_after_2s",
        "enemy_dot_lost_before_confirm",
        "qte_press_missed_window",
        "resource_target_disappeared_early",
        "resource_target_timeout",
        "item_pickup_fired_no_target",
    }
    assert snap_keys == fields
