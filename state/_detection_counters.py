"""Per-zone CV detection event counters.

Plain dataclass with `reset` / `snapshot` / `log_dump`. Used by main.py to
track patterns in CV behaviour (false positives, false negatives, timing
misses) over the lifetime of a single zone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectionCounters:
    zone_started_at: float = 0.0
    attack_no_healthbar_after_2s: int = 0
    enemy_dot_lost_before_confirm: int = 0
    qte_press_missed_window: int = 0
    resource_target_disappeared_early: int = 0
    resource_target_timeout: int = 0
    item_pickup_fired_no_target: int = 0

    def reset(self, now: float) -> None:
        self.zone_started_at = float(now)
        self.attack_no_healthbar_after_2s = 0
        self.enemy_dot_lost_before_confirm = 0
        self.qte_press_missed_window = 0
        self.resource_target_disappeared_early = 0
        self.resource_target_timeout = 0
        self.item_pickup_fired_no_target = 0

    def snapshot(self) -> dict:
        return {
            "attack_no_healthbar_after_2s": self.attack_no_healthbar_after_2s,
            "enemy_dot_lost_before_confirm": self.enemy_dot_lost_before_confirm,
            "qte_press_missed_window": self.qte_press_missed_window,
            "resource_target_disappeared_early": self.resource_target_disappeared_early,
            "resource_target_timeout": self.resource_target_timeout,
            "item_pickup_fired_no_target": self.item_pickup_fired_no_target,
        }

    def log_dump(self, now: float, prefix: str = "[CNT]") -> None:
        elapsed = now - self.zone_started_at
        parts = [f"zone={elapsed:.1f}s"]
        for k, v in self.snapshot().items():
            parts.append(f"{k}={v}")
        print(f"{prefix} " + " ".join(parts))
