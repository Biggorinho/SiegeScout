"""
Lightweight session stats: save-file character reader + packet-sniff tracker.

Sniffs Hero_Siege TCP traffic for:
  - currencyData  → session gold earned + gold/hr
  - satanicZoneName + buffs  → zone name (decoded) + active buff list
  - addedItemObject  → item drops by rarity
  - experience/level  → live character update

Requires Npcap on Windows + admin privileges for packet sniffing.
Gracefully degrades if Scapy/psutil are unavailable.

Module-level singleton — both gui.py and main.py share the same instance:
    import stats_tracker
    tracker = stats_tracker.get_tracker()     # None until init_tracker()
    tracker = stats_tracker.init_tracker(...)  # creates/replaces
"""
import configparser
import glob
import json
import logging
import os
import re
import threading
import time
from typing import Callable

# Suppress Scapy's "No libpcap provider available" noise before it's imported.
# The warning is logged through scapy.loading (not scapy.runtime) on Windows.
for _scapy_logger in ("scapy", "scapy.runtime", "scapy.loading", "scapy.contrib"):
    logging.getLogger(_scapy_logger).setLevel(logging.ERROR)

# ── class ID → name ──────────────────────────────────────────────────────────
_CLASS_NAMES: dict[int, str] = {
    0:  "Barbarian",   1:  "Wizard",    2:  "Ranger",   3:  "Witch Doctor",
    4:  "Paladin",     5:  "Monk",      6:  "Vampire",  7:  "Pirate",
    8:  "Necromancer", 9:  "Druid",     10: "Amazon",   11: "Cleric",
    12: "Rune Master", 13: "Runebinder",14: "Technomancer",
    15: "Spirit Warden", 16: "Spellblade",
}

# ── zone buff names (1-indexed list, matches hero-siege-stats satanic_buffs) ─
_SATANIC_BUFFS: list[str] = [
    "Loot Goblin I",    "Loot Goblin II",   "Rune Master",
    "Gold Hunger",      "Heroic Windfall",  "Angelic Fortune",
    "Zephys Grace",     "Fury of Tempest",  "Rapid Casting",
    "Onslaught",        "Nether Surge",     "Relic Keepers",
    "Goblin's Greed",   "Artifact Digger",  "Artifact Seeker",
    "Artifact Excavator","Recruit",          "Combat Training",
    "Battle Scarred",   "Clairvoyance",     "Aftermath",
    "Deep Cuts",        "Old Town",         "Terror Zone",
    "Fields of Carnage",
]

# Buffs that meaningfully change how the bot should behave
BUFF_CLAIRVOYANCE   = "Clairvoyance"    # +100% all recovery → flee less
BUFF_AFTERMATH      = "Aftermath"       # 3% chance to spawn Legion on death → enemies respawn
BUFF_ONSLAUGHT      = "Onslaught"       # +100% attack damage → enemies die faster
BUFF_ZEPHYS_GRACE   = "Zephys Grace"    # +50% movement speed
BUFF_LOOT_GOBLIN_I  = "Loot Goblin I"   # item drop bonus → auto-enable loot pickup
BUFF_LOOT_GOBLIN_II = "Loot Goblin II"  # item drop bonus → auto-enable loot pickup

# ── satanic zone name table (Act 1-8, 5 zones each, 1-indexed) ───────────────
_ZONE_NAMES: dict[int, list[str]] = {
    1: ["Outskirts of Inoya", "Fields of Battle", "The Pumpkin Patch",
        "Woodhill Plains", "King's Garden"],
    2: ["Crystal Village", "Chilling Lake", "Arctic Tundra",
        "Snowy Mountains", "The Glacial Trail"],
    3: ["Corrupted Oasis", "Dry Hills", "Mos'Arthim Desert",
        "Pyramid Level 1", "Pyramid Level 2"],
    4: ["Old Mining Village", "The Highland Mines", "Corrupted Cave",
        "The Nightmare", "The Devil's Breach"],
    5: ["Mt. Fuji", "Misty Swamp", "Fuji Coast",
        "Sea of Karponia", "Temple of Zamjo"],
    6: ["Highland Graveyard", "The Cathedral", "Prison Dungeon",
        "Steam Train", "The Depths of Hell"],
    7: ["Deep Space", "Event Horizon", "The Black Hole",
        "Parallel Dimension", "Subconscious Mind"],
    8: ["Forest of the Slain", "Flooded Plains", "Forgotten Caves",
        "Camp of Souls", "Hellheim"],
}

# ── item rarity labels ────────────────────────────────────────────────────────
_RARITY_NAMES: dict[int, str] = {
    0: "Normal", 1: "Magic", 2: "Rare", 3: "Heroic", 4: "Angelic",
}
# Abbreviations shown in the stats card
_RARITY_SHORT: dict[int, str] = {
    0: "N", 1: "M", 2: "R", 3: "H", 4: "A",
}

# currency mode keys present in every gold packet
_CURRENCY_KEYS = ("GSS", "GSH", "GNS", "GNH", "GBP")

# module-level singleton
_tracker: "SessionTracker | None" = None


def get_tracker() -> "SessionTracker | None":
    return _tracker


def init_tracker(
    on_zone_entered: "Callable[[str, list[str]], None] | None" = None,
    on_char_updated: "Callable[[dict], None] | None" = None,
    on_item_dropped: "Callable[[int], None] | None" = None,
    on_enemy_killed: "Callable[[int], None] | None" = None,
) -> "SessionTracker":
    """Create (or replace) the module-level SessionTracker and return it."""
    global _tracker
    if _tracker is not None:
        _tracker.stop()
    _tracker = SessionTracker(
        on_zone_entered=on_zone_entered,
        on_char_updated=on_char_updated,
        on_item_dropped=on_item_dropped,
        on_enemy_killed=on_enemy_killed,
    )
    return _tracker


# ── helpers ───────────────────────────────────────────────────────────────────

def _decode_zone(raw: str) -> str:
    """'SZ_3_2' → 'Act 3 : Dry Hills'.  Falls back to raw string."""
    try:
        parts = raw.split("_")          # ['SZ', '3', '2']
        act   = int(parts[1])
        idx   = int(parts[2]) - 1      # 0-based
        zones = _ZONE_NAMES.get(act)
        if zones and 0 <= idx < len(zones):
            return f"Act {act} : {zones[idx]}"
    except Exception:
        pass
    return raw


def _parse_buffs(buffs_str: str) -> list:
    """'10|20|23' → ['Onslaught', 'Clairvoyance', 'Old Town']"""
    result = []
    for token in buffs_str.split("|"):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token) - 1
            if 0 <= idx < len(_SATANIC_BUFFS):
                result.append(_SATANIC_BUFFS[idx])
        except ValueError:
            pass
    return result


# ── save-file character reader ────────────────────────────────────────────────

def _hs_save_dir() -> "str | None":
    appdata = os.environ.get("LOCALAPPDATA", "")
    for sub in (r"Hero_Siege\hs2saves", r"Hero_Siege\hseditor\saves",
                r"Hero_Siege\saves", r"Hero_Siege"):
        p = os.path.join(appdata, sub)
        if os.path.isdir(p):
            return p
    print(f"[STATS] save dir not found under LOCALAPPDATA={appdata!r}")
    return None


def read_active_character() -> "dict | None":
    """
    Read most-recently-modified Hero Siege save file.
    Returns {"name", "level", "herolevel", "class_name", "gold"} or None.
    """
    save_dir = _hs_save_dir()
    if not save_dir:
        return None
    files = glob.glob(os.path.join(save_dir, "herosiege*.hss"))
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    cfg = configparser.ConfigParser()
    try:
        cfg.read(latest, encoding="utf-8")
    except Exception:
        return None
    section = cfg["0"] if "0" in cfg else {}
    cid = int(section.get("class", -1))
    return {
        "name":       section.get("name", "Unknown"),
        "level":      int(section.get("level", 0)),
        "herolevel":  int(section.get("herolevel", 0)),
        "class_name": _CLASS_NAMES.get(cid, f"Class #{cid}" if cid >= 0 else "?"),
        "gold":       int(section.get("gold", 0)),
    }


# ── SessionTracker ────────────────────────────────────────────────────────────

class SessionTracker:
    """
    Single Scapy AsyncSniffer handling all Hero_Siege packet types:
      - currencyData   → gold earned + gold/hr
      - satanicZoneName + buffs  → decoded zone name + active buff list
      - addedItemObject → item drops by rarity
      - experience / level  → live character data update

    All callbacks fire from the sniffer thread.  Tkinter callers must
    use self.after(0, cb) to marshal back to the main thread.
    """

    def __init__(
        self,
        on_zone_entered: "Callable[[str, list[str]], None] | None" = None,
        on_char_updated: "Callable[[dict], None] | None" = None,
        on_item_dropped: "Callable[[int], None] | None" = None,
        on_enemy_killed: "Callable[[int], None] | None" = None,
    ):
        # ── gold
        self._gold_lock      = threading.Lock()
        self._gold_baseline: "int | None" = None
        self._gold_current:  "int | None" = None
        self._session_start: "float | None" = None

        # ── zone + buffs
        self._zone_lock       = threading.Lock()
        self._current_zone:   "str | None" = None
        self._zone_entered_at: "float | None" = None
        self._active_buffs:   list = []
        self._zone_event      = threading.Event()

        # ── items
        self._item_lock       = threading.Lock()
        self._items_by_rarity: "dict[int, int]" = {}

        # ── live character (from account packets)
        self._char_lock    = threading.Lock()
        self._live_char:   "dict | None" = None   # keys: name, level, herolevel, class_name
        _saved = read_active_character()
        self._my_char_name: "str | None" = _saved["name"] if _saved else None

        # ── kills (EXP delta detection)
        self._exp_lock     = threading.Lock()
        self._prev_exp:    "int | None" = None
        self._session_exp:  int = 0
        self._kill_count:   int = 0
        self._last_kill_t:  float = 0.0

        # ── callbacks
        self._on_zone_entered = on_zone_entered  # (zone_name, buff_names)
        self._on_char_updated = on_char_updated  # (char_dict)
        self._on_item_dropped = on_item_dropped  # (rarity_int)
        self._on_enemy_killed = on_enemy_killed  # (exp_delta_int)

        self._sniffer = None
        self._poll_thread: "threading.Thread | None" = None
        self._poll_stop:   "threading.Event | None" = None
        self.available = False
        try:
            import psutil       # noqa: F401
            import scapy.all    # noqa: F401
            # On Windows, Scapy imports successfully even without Npcap (with a warning).
            # Verify wpcap.dll is actually present before marking the sniffer available,
            # otherwise AsyncSniffer.start() prints the pcap warning into the GUI log.
            import sys as _sys
            if _sys.platform == "win32":
                import ctypes as _ct
                _ct.WinDLL("wpcap.dll")   # raises OSError if Npcap/WinPcap not installed
            self.available = True
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        """Reset gold baseline and (re)start the sniffer (or save-file poller fallback)."""
        _saved = read_active_character()
        if _saved:
            self._my_char_name = _saved["name"]
        with self._gold_lock:
            self._gold_baseline = None
            self._gold_current  = None
            self._session_start = time.time()
        with self._item_lock:
            self._items_by_rarity.clear()
        with self._exp_lock:
            self._prev_exp    = None
            self._session_exp = 0
            self._kill_count  = 0
            self._last_kill_t = 0.0
        # Always start save-file poller for gold fallback.
        # If the sniffer also runs it will update gold_current as well (whichever is higher wins).
        self._start_savefile_poller()
        if self.available:
            self._launch_sniffer()

    def stop(self):
        sniffer, self._sniffer = self._sniffer, None
        if sniffer:
            try:
                sniffer.stop()
            except Exception:
                pass
        if self._poll_stop:
            self._poll_stop.set()
        self._poll_thread = None
        self._poll_stop   = None

    # ── save-file gold fallback (no Npcap) ───────────────────────────────────

    @property
    def gold_source(self) -> str:
        """Returns 'packets', 'savefile', or 'unavailable'."""
        if self._poll_thread and self._poll_thread.is_alive():
            return "packets" if self.available else "savefile"
        return "unavailable"

    def _start_savefile_poller(self):
        if self._poll_stop:
            self._poll_stop.set()
        self._poll_stop   = threading.Event()
        self._poll_thread = threading.Thread(
            target=self._savefile_poll_loop,
            args=(self._poll_stop,),
            daemon=True,
        )
        self._poll_thread.start()

    def _savefile_poll_loop(self, stop_event: threading.Event):
        while True:
            try:
                char = read_active_character()
                if char is not None:
                    total = int(char.get("gold", 0))
                    with self._gold_lock:
                        if self._gold_baseline is None:
                            self._gold_baseline = total
                        if self._gold_current is None or total >= self._gold_current:
                            self._gold_current = total
            except Exception:
                pass
            if stop_event.wait(6.0):
                break

    def wait_for_zone_entry(self, timeout: float, since_t: float) -> "str | None":
        """
        Block until a satanicZoneName packet with timestamp > since_t arrives,
        or until timeout elapses.  Returns decoded zone name or None.
        """
        deadline = since_t + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._zone_event.wait(timeout=min(0.25, remaining))
            self._zone_event.clear()
            with self._zone_lock:
                if (self._zone_entered_at is not None
                        and self._zone_entered_at > since_t):
                    return self._current_zone
        return None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def current_zone(self) -> "str | None":
        with self._zone_lock:
            return self._current_zone

    @property
    def zone_entered_at(self) -> "float | None":
        with self._zone_lock:
            return self._zone_entered_at

    @property
    def active_buffs(self) -> list:
        """Returns a snapshot of the buff names currently active in this zone."""
        with self._zone_lock:
            return list(self._active_buffs)

    @property
    def items_by_rarity(self) -> dict:
        """Returns {rarity_int: count} for items collected this session."""
        with self._item_lock:
            return dict(self._items_by_rarity)

    @property
    def live_char(self) -> "dict | None":
        with self._char_lock:
            return dict(self._live_char) if self._live_char else None

    @property
    def kills(self) -> int:
        with self._exp_lock:
            return self._kill_count

    @property
    def kills_per_hour(self) -> int:
        with self._exp_lock:
            _kc = self._kill_count
        if _kc == 0:
            return 0
        with self._gold_lock:
            _start = self._session_start
        if _start is None:
            return 0
        elapsed = time.time() - _start
        if elapsed < 30:
            return 0
        return int(_kc / elapsed * 3600)

    @property
    def exp_earned(self) -> int:
        with self._exp_lock:
            return self._session_exp

    @property
    def last_kill_t(self) -> float:
        with self._exp_lock:
            return self._last_kill_t

    @property
    def gold_earned(self) -> int:
        with self._gold_lock:
            if self._gold_baseline is None or self._gold_current is None:
                return 0
            return max(0, self._gold_current - self._gold_baseline)

    @property
    def gold_per_hour(self) -> int:
        with self._gold_lock:
            if (self._session_start is None
                    or self._gold_baseline is None
                    or self._gold_current is None):
                return 0
            elapsed = time.time() - self._session_start
            if elapsed < 30:
                return 0
            return int(max(0, self._gold_current - self._gold_baseline) / elapsed * 3600)

    # ── internals ─────────────────────────────────────────────────────────────

    def _launch_sniffer(self):
        if self._sniffer is not None:
            try:
                self._sniffer.stop()
            except Exception:
                pass
        import io as _io, sys as _sys, ctypes as _ct
        _old_stderr = _sys.stderr
        _is_admin = bool(_ct.windll.shell32.IsUserAnAdmin()) if _sys.platform == "win32" else True
        try:
            from scapy.all import AsyncSniffer
            ips = self._get_hs_ips()
            pkt_filter = (
                f"(host {' or host '.join(ips)}) and len > 30"
                if ips else "tcp and len > 30"
            )
            print(f"[STATS] Starting sniffer | admin={_is_admin} | ips={ips or 'any'} | filter: {pkt_filter}")
            self._sniffer = AsyncSniffer(
                filter=pkt_filter, prn=self._on_packet, store=False)
            _sys.stderr = _io.StringIO()
            try:
                self._sniffer.start()
            finally:
                _sys.stderr = _old_stderr
            print("[STATS] Sniffer running")
        except Exception as e:
            _sys.stderr = _old_stderr
            print(f"[STATS] Sniffer failed ({e}); save-file fallback only")
            self.available = False

    @staticmethod
    def _get_hs_ips() -> set:
        try:
            import psutil
            pids = {
                p.info["pid"]
                for p in psutil.process_iter(["pid", "name"])
                if "hero_siege" in (p.info.get("name") or "").lower()
            }
            return {
                c.raddr.ip
                for c in psutil.net_connections("inet")
                if c.pid in pids and c.raddr
            }
        except Exception:
            return set()

    def _on_packet(self, packet):
        try:
            from scapy.layers.inet import IP, TCP
            if IP not in packet or TCP not in packet:
                return
            try:
                raw_bytes = bytes(packet.load)
            except AttributeError:
                return

            raw = raw_bytes.decode("utf-8", errors="replace")
            m = re.search(r"\{.+\}", raw)
            if not m:
                return
            try:
                msg = json.loads(m.group(0))
            except json.JSONDecodeError:
                return
            if not isinstance(msg, dict):
                return
            if "steam" in msg or "inventory_charms" in msg:
                return

            # Log each unique key-set once so we can discover new packet types.
            # Each unique "shape" is printed exactly once during the session.
            if not hasattr(self, "_seen_shapes"):
                self._seen_shapes: set = set()
                self._shape_n: int = 0
            _sig = frozenset(msg.keys())
            if _sig not in self._seen_shapes and self._shape_n < 30:
                self._seen_shapes.add(_sig)
                self._shape_n += 1
                _pv: dict = {}
                for _k in sorted(msg.keys()):
                    _v = msg[_k]
                    if isinstance(_v, (bool, int, float)):
                        _pv[_k] = _v
                    elif isinstance(_v, str):
                        _pv[_k] = (_v[:48] + "…") if len(_v) > 48 else _v
                    elif isinstance(_v, dict):
                        _pv[_k] = f"{{…{list(_v.keys())[:5]}}}"
                    elif isinstance(_v, list):
                        _pv[_k] = f"[…{len(_v)}]"
                print(f"[STATS] shape#{self._shape_n}: {_pv}")

            # ── gold ─────────────────────────────────────────────────────────
            cd = msg.get("currencyData") or msg.get("currency_data")
            if isinstance(cd, dict):
                total = sum(int(cd.get(k, 0)) for k in _CURRENCY_KEYS)
                with self._gold_lock:
                    if self._gold_baseline is None:
                        self._gold_baseline = total
                    if self._gold_current is None or total >= self._gold_current:
                        self._gold_current = total

            # ── zone + buffs ──────────────────────────────────────────────────
            raw_zone = msg.get("satanicZoneName")
            if raw_zone and isinstance(raw_zone, str):
                zone_name = _decode_zone(raw_zone.strip())
                raw_buffs  = msg.get("buffs", "")
                buff_names = _parse_buffs(str(raw_buffs)) if raw_buffs else []
                with self._zone_lock:
                    self._current_zone    = zone_name
                    self._zone_entered_at = time.time()
                    self._active_buffs    = buff_names
                self._zone_event.set()
                if self._on_zone_entered:
                    try:
                        self._on_zone_entered(zone_name, buff_names)
                    except Exception:
                        pass

            # ── item drops ───────────────────────────────────────────────────
            # Flat item objects (single item per packet).
            for _ifield in ("addedItemObject", "droppedItem", "newItem",
                            "lootItem", "itemDrop", "item"):
                _iobj = msg.get(_ifield)
                if not isinstance(_iobj, dict):
                    continue
                for _rfield in ("rarity", "itemRarity", "quality", "tier", "grade"):
                    if _rfield in _iobj:
                        try:
                            rarity = int(_iobj[_rfield])
                            with self._item_lock:
                                self._items_by_rarity[rarity] = \
                                    self._items_by_rarity.get(rarity, 0) + 1
                            if self._on_item_dropped:
                                try:
                                    self._on_item_dropped(rarity)
                                except Exception:
                                    pass
                        except (ValueError, TypeError):
                            pass
                        break
                break

            # Dict-of-items packets: itemData / itemsAdded → {item_id: item_obj, ...}
            if not hasattr(self, "_item_val_logged"):
                self._item_val_logged: set = set()
            for _ifield in ("itemData", "itemsAdded"):
                _idict = msg.get(_ifield)
                if not isinstance(_idict, dict) or not _idict:
                    continue
                # One-time debug: show the raw type + content of one item value.
                if _ifield not in self._item_val_logged:
                    self._item_val_logged.add(_ifield)
                    _sk = next(iter(_idict))
                    _sv = _idict[_sk]
                    print(f"[STATS-ITEM] {_ifield} key={_sk!r} val_type={type(_sv).__name__} val={str(_sv)[:200]!r}")
                for _iobj in _idict.values():
                    # Values might be JSON-encoded strings — try to decode them.
                    if isinstance(_iobj, str):
                        try:
                            _iobj = json.loads(_iobj)
                        except Exception:
                            pass
                    if not isinstance(_iobj, dict):
                        continue
                    for _rfield in ("rarity", "itemRarity", "quality", "tier", "grade"):
                        if _rfield in _iobj:
                            try:
                                rarity = int(_iobj[_rfield])
                                with self._item_lock:
                                    self._items_by_rarity[rarity] = \
                                        self._items_by_rarity.get(rarity, 0) + 1
                                if self._on_item_dropped:
                                    try:
                                        self._on_item_dropped(rarity)
                                    except Exception:
                                        pass
                            except (ValueError, TypeError):
                                pass
                            break

            # Character is read from the save file at bot start — packet-based
            # player data comes from global server broadcasts (other players online),
            # not from our own character, so we ignore it here.

            # ── exp / kills ───────────────────────────────────────────────────
            # 1) Numeric exp field variants.
            _exp_raw: "int | None" = None
            for _efield in ("experience", "exp", "xp", "totalExp", "currentExp"):
                _ev = msg.get(_efield)
                if _ev is not None:
                    try:
                        _exp_raw = int(_ev)
                        break
                    except (ValueError, TypeError):
                        pass

            if _exp_raw is not None:
                _fire_kill: "int | None" = None
                with self._exp_lock:
                    if self._prev_exp is not None and _exp_raw > self._prev_exp:
                        delta = _exp_raw - self._prev_exp
                        if delta < 5_000_000:
                            self._session_exp += delta
                            self._kill_count  += 1
                            self._last_kill_t  = time.time()
                            _fire_kill = delta
                    self._prev_exp = _exp_raw
                if _fire_kill is not None and self._on_enemy_killed:
                    try:
                        self._on_enemy_killed(_fire_kill)
                    except Exception:
                        pass

            # "N xp added." — guild XP sync, fires once per zone. Add to session
            # exp total so the GUI can display cumulative exp earned.
            _msg_str = msg.get("message", "")
            if isinstance(_msg_str, str) and "xp added" in _msg_str.lower():
                _xm = re.search(r"(\d+)\s*xp\s*added", _msg_str, re.IGNORECASE)
                if _xm:
                    try:
                        _msg_xp = int(_xm.group(1))
                        if 0 < _msg_xp < 500_000_000:
                            _fire_xp: "int | None" = None
                            with self._exp_lock:
                                self._session_exp += _msg_xp
                                _fire_xp = _msg_xp
                            if _fire_xp is not None and self._on_enemy_killed:
                                try:
                                    self._on_enemy_killed(_fire_xp)
                                except Exception:
                                    pass
                    except Exception:
                        pass

        except Exception:
            pass
