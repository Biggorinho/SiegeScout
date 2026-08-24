"""
SiegeScouter GUI — native Tkinter (no extra deps)
Drop-in replacement for gui.py. Works on any Python version with tkinter.

Optional: patch main.run() to accept `on_vote_reset=callback` — the GUI
will bump the reset counter each time it fires. If you don't patch, the
counter will stay at 0 (everything else works).
"""

import io
import inspect
import json
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

import config
import license as _lic
import stats_tracker as _stats


# ── palette (SiegeScout) ─────────────────────────────────────────────────────
BG = "#1a1c1a"
SURFACE = "#232623"
BORDER = "#363a35"
FG = "#d4cfa0"
FG_MUTE = "#7a7d6a"
BLUE = "#5b9bd5"
GREEN = "#5aaa46"
RED = "#c94040"
YELLOW = "#d4aa40"
MAUVE = "#9b7fd4"
PEACH = "#c87d3a"
LOG_BG = "#111311"

LOGO_FILE = os.path.join("assets", "logo.png")
_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


def _asset(name: str) -> str:
    for base in (
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(sys.executable),
        ".",
    ):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return name


class _QueueWriter(io.TextIOBase):
    """Captures writes into the GUI queue AND (optionally) appends them to
    a log file with ISO-8601 timestamp prefixes for downstream analysis."""

    def __init__(self, q: queue.Queue, log_file=None):
        self._q = q
        self._log_file = log_file
        self._line_start = True

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._q.put(text)
        if self._log_file is not None:
            try:
                self._write_to_log(text)
            except OSError:
                # Disk error — disable file logging silently to avoid spamming
                # the GUI; the in-memory queue still works.
                self._log_file = None
        return len(text)

    def _write_to_log(self, text: str):
        # Prefix each new line with an ISO-8601 timestamp.
        import datetime
        i = 0
        while i < len(text):
            if self._line_start:
                ts = datetime.datetime.now().isoformat(timespec="milliseconds")
                self._log_file.write(f"{ts}  ")
                self._line_start = False
            nl = text.find("\n", i)
            if nl < 0:
                self._log_file.write(text[i:])
                break
            self._log_file.write(text[i:nl + 1])
            self._line_start = True
            i = nl + 1
        self._log_file.flush()

    def flush(self):
        return None


class HoverButton(tk.Label):
    """A tk.Label styled as a pill button with hover + click."""

    def __init__(self, parent, text, command, bg, fg, hover_bg=None, pady=10, padx=16, font=None):
        super().__init__(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            font=font,
            padx=padx,
            pady=pady,
            cursor="hand2",
        )
        self._bg = bg
        self._hover = hover_bg or bg
        self._cmd = command
        self.bind("<Enter>", lambda e: self.config(bg=self._hover))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))
        self.bind("<Button-1>", lambda e: self._cmd() if self._cmd else None)

    def set_style(self, text, bg, fg, hover_bg):
        self._bg = bg
        self._hover = hover_bg
        self.config(text=text, bg=bg, fg=fg)


class PulseDot(tk.Canvas):
    """Small colored dot that pulses while alive."""

    def __init__(self, parent, color=GREEN, size=10):
        super().__init__(
            parent,
            width=size + 4,
            height=size + 4,
            bg=parent.cget("bg"),
            highlightthickness=0,
        )
        self._size = size
        self._color = color
        self._alive = False
        self._phase = 0
        self._item = self.create_oval(2, 2, size + 2, size + 2, fill=color, outline="")
        self._halo = self.create_oval(2, 2, size + 2, size + 2, outline=color, width=1)

    def set(self, alive, color=None):
        self._alive = alive
        if color:
            self._color = color
        self.itemconfig(self._item, fill=self._color)
        self.itemconfig(self._halo, outline=self._color)
        if alive:
            self._tick()

    def _tick(self):
        if not self._alive:
            self.coords(self._halo, 2, 2, self._size + 2, self._size + 2)
            return
        self._phase = (self._phase + 1) % 30
        grow = self._phase / 30.0 * 6
        x0 = 2 - grow
        y0 = 2 - grow
        x1 = self._size + 2 + grow
        y1 = self._size + 2 + grow
        self.coords(self._halo, x0, y0, x1, y1)
        self.after(50, self._tick)


class Switch(tk.Canvas):
    """iOS-style toggle switch."""

    def __init__(self, parent, initial=False, disabled=False, onchange=None):
        super().__init__(
            parent,
            width=36,
            height=22,
            bg=parent.cget("bg"),
            highlightthickness=0,
            cursor="arrow" if disabled else "hand2",
        )
        self._on = bool(initial)
        self._disabled = disabled
        self._cb = onchange
        self._track = self.create_rectangle(0, 2, 36, 20, fill=self._track_color(), outline="")
        self._knob = self.create_oval(2, 3, 18, 19, fill=FG, outline="")
        self._redraw()
        if not disabled:
            self.bind("<Button-1>", self._toggle)

    def _track_color(self):
        if self._disabled:
            return "#3c3c50"
        return GREEN if self._on else "#3c3c50"

    def _redraw(self):
        self.itemconfig(self._track, fill=self._track_color())
        x = 16 if self._on else 0
        self.coords(self._knob, 2 + x, 3, 18 + x, 19)
        self.itemconfig(self._knob, fill=BG if self._on else FG)
        if self._disabled:
            self.itemconfig(self._knob, fill="#5c5c70")

    def _toggle(self, _):
        self._on = not self._on
        self._redraw()
        if self._cb:
            self._cb(self._on)

    def get(self):
        return self._on

    def set(self, value: bool):
        if self._disabled:
            return
        self._on = bool(value)
        self._redraw()


class BotUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SiegeScout")
        self.configure(bg=BG)
        self.resizable(False, False)
        try:
            self.tk.call("tk", "scaling", 1.1)
        except Exception:
            pass

        self._stop_event: threading.Event | None = None
        self._bot_thread: threading.Thread | None = None
        self._log_queue: queue.Queue = queue.Queue()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._reset_count = 0
        self._license_valid       = False
        self._license_error_lbl   = None   # tk.Label | None
        self._license_entry        = None   # tk.Entry | None
        self._license_activate_btn = None   # HoverButton | None
        self._cta_container        = None   # tk.Frame | None
        self._runtime_sec = 0
        self._runtime_job = None
        self._tracker = _stats.init_tracker(
            on_zone_entered=self._on_zone_entered_packet,
            on_char_updated=self._on_char_updated_packet,
            on_item_dropped=self._on_item_dropped_packet,
            on_enemy_killed=self._on_kill_packet,
        )
        self._stats_job: str | None = None
        self._char_lbl: tk.Label | None = None
        self._zone_lbl: tk.Label | None = None
        self._buffs_lbl: tk.Label | None = None
        self._items_lbl: tk.Label | None = None
        self._kills_lbl: tk.Label | None = None
        self._gold_earned_lbl: tk.Label | None = None
        self._gold_hr_lbl: tk.Label | None = None
        self._gold_src_lbl: tk.Label | None = None

        self.f_ui = tkfont.Font(family="Segoe UI", size=10)
        self.f_ui_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.f_title = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.f_mono = tkfont.Font(family="Consolas", size=9)
        self.f_mono_bold = tkfont.Font(family="Consolas", size=9, weight="bold")
        self.f_cta = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.f_huge = tkfont.Font(family="Segoe UI", size=26, weight="bold")
        self.f_cap = tkfont.Font(family="Consolas", size=8, weight="bold")

        self._build_ui()
        self._load_settings()
        self.report_callback_exception = self._on_tk_exception
        self.after(80, self._poll_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.geometry("560x980")
        self.after(50, self._init_license)

    def _build_ui(self):
        pad = 14
        root = tk.Frame(self, bg=BG)
        root.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        header = tk.Frame(root, bg=BG)
        header.pack(fill=tk.X)

        brand = tk.Frame(header, bg=BG)
        brand.pack(side=tk.LEFT)

        try:
            _img = Image.open(_asset(LOGO_FILE)).resize((48, 48), Image.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(_img)
            logo = tk.Label(brand, image=self._logo_img, bg=BG, bd=0)
        except Exception:
            logo = tk.Label(brand, text="SS", bg=BLUE, fg=BG, font=self.f_title, padx=8, pady=4)
        logo.pack(side=tk.LEFT, padx=(0, 10))

        brand_text = tk.Frame(brand, bg=BG)
        brand_text.pack(side=tk.LEFT)
        tk.Label(brand_text, text="SiegeScout", bg=BG, fg=FG, font=self.f_title).pack(anchor=tk.W)
        tk.Label(brand_text, text="by Biggorinho · v1.4", bg=BG, fg=FG_MUTE, font=("Consolas", 8)).pack(anchor=tk.W)

        status = tk.Frame(header, bg=BG)
        status.pack(side=tk.RIGHT, pady=4)
        self._pulse = PulseDot(status, color=FG_MUTE)
        self._pulse.pack(side=tk.LEFT, padx=(0, 4))
        self._status_lbl = tk.Label(status, text="Idle", bg=BG, fg=FG_MUTE, font=self.f_ui_bold)
        self._status_lbl.pack(side=tk.LEFT)

        # ── setup notice ─────────────────────────────────────────────────────
        notice = tk.Frame(root, bg="#192519", padx=12, pady=7,
                          highlightbackground="#2d4d2d", highlightthickness=1)
        notice.pack(fill=tk.X, pady=(12, 0))
        tk.Label(notice, text="Play in 1920×1080 Windowed Fullscreen",
                 bg="#192519", fg="#72aa72", font=("Consolas", 8)).pack(anchor=tk.W)
        tk.Label(notice, text="Start the bot from the Zone Selection screen",
                 bg="#192519", fg="#72aa72", font=("Consolas", 8)).pack(anchor=tk.W)

        cta_row = tk.Frame(root, bg=BG)
        cta_row.pack(fill=tk.X, pady=(12, 0))

        self._cta_container = tk.Frame(cta_row, bg=BG)
        self._cta_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._cta = HoverButton(
            self._cta_container,
            text="▶  Start Bot",
            command=self._toggle_bot,
            bg=GREEN,
            fg="#14341b",
            hover_bg="#7fd46f",
            pady=12,
            padx=16,
            font=self.f_cta,
        )
        self._cta.pack(fill=tk.X, expand=True)

        reset_card = tk.Frame(cta_row, bg=SURFACE, padx=14, pady=8, highlightbackground=BORDER, highlightthickness=1)
        reset_card.pack(side=tk.LEFT, padx=(10, 0))
        tk.Label(reset_card, text="RESETS", bg=SURFACE, fg=FG_MUTE, font=self.f_cap).pack(anchor=tk.W)
        self._reset_lbl = tk.Label(reset_card, text="0", bg=SURFACE, fg=MAUVE, font=self.f_huge)
        self._reset_lbl.pack(anchor=tk.W, pady=(2, 0))
        tk.Label(reset_card, text="session", bg=SURFACE, fg=FG_MUTE, font=("Consolas", 8)).pack(anchor=tk.W)

        # ── session stats card ────────────────────────────────────────────────
        stats_card = tk.Frame(root, bg=SURFACE, padx=14, pady=10,
                              highlightbackground=BORDER, highlightthickness=1)
        stats_card.pack(fill=tk.X, pady=(10, 0))

        char_row = tk.Frame(stats_card, bg=SURFACE)
        char_row.pack(fill=tk.X)
        tk.Label(char_row, text="CHAR", bg=SURFACE, fg=FG_MUTE, font=self.f_cap).pack(side=tk.LEFT)
        self._char_lbl = tk.Label(char_row, text="—", bg=SURFACE, fg=FG, font=self.f_ui)
        self._char_lbl.pack(side=tk.LEFT, padx=(8, 0))

        zone_row = tk.Frame(stats_card, bg=SURFACE)
        zone_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(zone_row, text="ZONE", bg=SURFACE, fg=FG_MUTE, font=self.f_cap).pack(side=tk.LEFT)
        self._zone_lbl = tk.Label(zone_row, text="—", bg=SURFACE, fg=BLUE, font=self.f_ui)
        self._zone_lbl.pack(side=tk.LEFT, padx=(8, 0))

        buffs_row = tk.Frame(stats_card, bg=SURFACE)
        buffs_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(buffs_row, text="BUFFS", bg=SURFACE, fg=FG_MUTE, font=self.f_cap).pack(side=tk.LEFT)
        self._buffs_lbl = tk.Label(buffs_row, text="—", bg=SURFACE, fg=MAUVE, font=self.f_ui)
        self._buffs_lbl.pack(side=tk.LEFT, padx=(8, 0))

        tk.Frame(stats_card, bg=BORDER, height=1).pack(fill=tk.X, pady=(6, 0))

        # ── 2×2 stat grid: KILLS | ITEMS  /  GOLD | RATE ─────────────────────
        def _stat_cell(parent, label, color):
            cell = tk.Frame(parent, bg=SURFACE)
            cell.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(cell, text=label, bg=SURFACE, fg=FG_MUTE, font=self.f_cap).pack(side=tk.LEFT)
            val = tk.Label(cell, text="—", bg=SURFACE, fg=color, font=self.f_ui)
            val.pack(side=tk.LEFT, padx=(8, 0))
            return val

        row_a = tk.Frame(stats_card, bg=SURFACE)
        row_a.pack(fill=tk.X, pady=(6, 0))
        self._kills_lbl = _stat_cell(row_a, "EXP", GREEN)
        self._items_lbl = _stat_cell(row_a, "ITEMS", FG)

        row_b = tk.Frame(stats_card, bg=SURFACE)
        row_b.pack(fill=tk.X, pady=(4, 0))
        self._gold_earned_lbl = _stat_cell(row_b, "GOLD", YELLOW)
        self._gold_hr_lbl     = _stat_cell(row_b, "RATE", FG_MUTE)

        self._gold_src_lbl = tk.Label(stats_card, text="", bg=SURFACE, fg=FG_MUTE,
                                      font=("Consolas", 7))
        self._gold_src_lbl.pack(anchor=tk.W, pady=(2, 0))

        # ── Npcap notice (shown only when packet sniffing is unavailable) ─────
        if not self._tracker.available:
            npcap_frame = tk.Frame(root, bg="#2a1a0a", padx=12, pady=8,
                                   highlightbackground="#6b3d10", highlightthickness=1)
            npcap_frame.pack(fill=tk.X, pady=(10, 0))
            tk.Label(npcap_frame, text="Npcap not detected — live stats unavailable",
                     bg="#2a1a0a", fg=PEACH, font=("Consolas", 8, "bold")).pack(anchor=tk.W)
            tk.Label(npcap_frame,
                     text="Install Npcap for zone buffs, kills & item tracking:",
                     bg="#2a1a0a", fg=FG_MUTE, font=("Consolas", 8)).pack(anchor=tk.W, pady=(2, 0))
            link = tk.Label(npcap_frame, text="  npcap.com  →  Download",
                            bg="#2a1a0a", fg=BLUE, font=("Consolas", 8, "underline"), cursor="hand2")
            link.pack(anchor=tk.W)
            link.bind("<Button-1>", lambda _e: __import__("webbrowser").open("https://npcap.com/#download"))
            tk.Label(npcap_frame,
                     text="After install: restart SiegeScout as Administrator",
                     bg="#2a1a0a", fg=FG_MUTE, font=("Consolas", 8)).pack(anchor=tk.W, pady=(2, 0))

        opt = tk.Frame(root, bg=SURFACE, padx=14, pady=12, highlightbackground=BORDER, highlightthickness=1)
        opt.pack(fill=tk.X, pady=(14, 0))
        tk.Label(opt, text="OPTIONS", bg=SURFACE, fg=FG_MUTE, font=self.f_cap).pack(anchor=tk.W, pady=(0, 8))

        def _row(parent, label, default, enabled=True, soon=False):
            row = tk.Frame(parent, bg=SURFACE)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, bg=SURFACE, fg=FG if enabled else FG_MUTE, font=self.f_ui).pack(side=tk.LEFT)
            if soon:
                tag = tk.Label(row, text=" SOON ", bg="#3a2a1a", fg=PEACH, font=self.f_cap, padx=4)
                tag.pack(side=tk.LEFT, padx=(8, 0))
            sw = Switch(row, initial=default, disabled=not enabled)
            sw.pack(side=tk.RIGHT)
            return sw

        self._death_sw = _row(opt, "Enable Death Handling", getattr(config, "ENABLE_DEATH_HANDLING", True))
        self._melee_sw = _row(opt, "Melee Mode  (advances into enemies, no kiting)", getattr(config, "MELEE_MODE", False))
        self._mining_sw = _row(opt, "Enable Mining", getattr(config, "ENABLE_MINING", False))
        self._loot_sw = _row(opt, "Enable Loot Items", getattr(config, "ENABLE_LOOT_ITEMS", False))
        self._debug_sw = _row(opt, "Debug Snapshots", getattr(config, "ENABLE_DEBUG_SNAPSHOTS", False))

        tk.Frame(opt, bg=BORDER, height=1).pack(fill=tk.X, pady=(8, 4))

        zone_row = tk.Frame(opt, bg=SURFACE)
        zone_row.pack(fill=tk.X, pady=3)
        tk.Label(zone_row, text="Target Zone", bg=SURFACE, fg=FG, font=self.f_ui).pack(side=tk.LEFT)

        _zone_names = ["Satanic (auto)"] + sorted(config.SATANIC_FIXED_NODES.keys())
        self._zone_var = tk.StringVar(value="Satanic (auto)")
        zone_menu = tk.OptionMenu(zone_row, self._zone_var, *_zone_names)
        zone_menu.config(
            bg=SURFACE, fg=FG, activebackground=BLUE, activeforeground=FG,
            relief=tk.FLAT, bd=0, font=self.f_ui, highlightthickness=0,
            width=22, anchor="e",
        )
        zone_menu["menu"].config(
            bg=SURFACE, fg=FG, activebackground=BLUE, activeforeground=FG,
            font=self.f_ui, relief=tk.FLAT, bd=0,
        )
        zone_menu.pack(side=tk.RIGHT)

        # ── skill bindings ────────────────────────────────────────────────────
        sk = tk.Frame(root, bg=SURFACE, padx=14, pady=10,
                      highlightbackground=BORDER, highlightthickness=1)
        sk.pack(fill=tk.X, pady=(10, 0))
        tk.Label(sk, text="SKILL BINDINGS", bg=SURFACE, fg=FG_MUTE, font=self.f_cap).pack(anchor=tk.W, pady=(0, 6))

        def _skill_row(parent, label, key_hint, default_enabled, default_cd):
            row = tk.Frame(parent, bg=SURFACE)
            row.pack(fill=tk.X, pady=2)
            sw = Switch(row, initial=default_enabled)
            sw.pack(side=tk.LEFT)
            tk.Label(row, text=label, bg=SURFACE, fg=FG, font=self.f_ui).pack(side=tk.LEFT, padx=(8, 0))
            tk.Label(row, text=f" {key_hint} ", bg="#1e2e1e", fg=GREEN,
                     font=self.f_cap, padx=3).pack(side=tk.LEFT, padx=(6, 0))
            cd_var = tk.StringVar(value=str(default_cd))
            cd_entry = tk.Entry(row, textvariable=cd_var, width=5,
                                bg="#2a2e2a", fg=FG, insertbackground=FG,
                                relief=tk.FLAT, bd=2, font=self.f_ui, justify=tk.CENTER)
            cd_entry.pack(side=tk.RIGHT)
            tk.Label(row, text="s   CD:", bg=SURFACE, fg=FG_MUTE, font=self.f_ui).pack(side=tk.RIGHT, padx=(0, 2))
            return sw, cd_var

        self._rc_sw,  self._rc_cd  = _skill_row(sk, "Buff / Shield",    "KEY E", True, 10.0)
        self._k3_sw,  self._k3_cd  = _skill_row(sk, "Flasks",           "KEY 3", True, 50.0)
        self._mv_sw,  self._mv_cd  = _skill_row(sk, "Movement Ability", "KEY R", True,  3.0)

        log_card = tk.Frame(root, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        log_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        log_head = tk.Frame(log_card, bg=SURFACE)
        log_head.pack(fill=tk.X, padx=12, pady=(10, 8))
        tk.Label(log_head, text="LOG", bg=SURFACE, fg=FG_MUTE, font=self.f_cap).pack(side=tk.LEFT)
        self._log_count = tk.Label(log_head, text="0 lines", bg=SURFACE, fg=BLUE, font=self.f_cap)
        self._log_count.pack(side=tk.LEFT, padx=(8, 0))
        clear = tk.Label(log_head, text="clear ⌫", bg=SURFACE, fg=FG_MUTE, font=self.f_ui, cursor="hand2")
        clear.pack(side=tk.RIGHT)
        clear.bind("<Button-1>", lambda e: self._clear_log())

        log_frame = tk.Frame(log_card, bg=LOG_BG)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 0))

        self._log = tk.Text(
            log_frame,
            height=12,
            bg=LOG_BG,
            fg=FG,
            font=self.f_mono,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=8,
            wrap=tk.NONE,
            state=tk.DISABLED,
            insertbackground=FG,
        )
        self._log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(log_frame, command=self._log.yview, bg=SURFACE, troughcolor=LOG_BG, bd=0, highlightthickness=0)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.config(yscrollcommand=sb.set)

        self._log.tag_config("time", foreground=FG_MUTE)
        self._log.tag_config("info", foreground=BLUE)
        self._log.tag_config("ok", foreground=GREEN)
        self._log.tag_config("warn", foreground=YELLOW)
        self._log.tag_config("err", foreground=RED)
        self._log.tag_config("reset", foreground=MAUVE, font=self.f_mono_bold)

        footer = tk.Frame(root, bg=BG)
        footer.pack(fill=tk.X, pady=(10, 0))
        self._runtime_lbl = tk.Label(footer, text="runtime · 00:00:00", bg=BG, fg=FG_MUTE, font=("Consolas", 9))
        self._runtime_lbl.pack(side=tk.LEFT)
        tk.Label(footer, text="emergency stop  F8", bg=BG, fg=FG_MUTE, font=("Consolas", 9)).pack(side=tk.RIGHT)

    def _init_license(self):
        import threading
        key = _lic.get_saved_key()
        if key is None:
            self._show_license_panel()
        else:
            self._cta.set_style("Checking license…", bg=SURFACE, fg=FG_MUTE, hover_bg=SURFACE)
            threading.Thread(
                target=lambda: self.after(0, self._on_license_result, *_lic.validate_license(key)),
                daemon=True,
            ).start()

    def _show_license_panel(self):
        self._cta.pack_forget()
        panel = tk.Frame(self._cta_container, bg=BG)
        panel.pack(fill=tk.X)
        tk.Label(panel, text="Enter your SiegeScout license key",
                 bg=BG, fg=FG_MUTE, font=self.f_ui).pack(anchor=tk.W, pady=(0, 4))
        row = tk.Frame(panel, bg=BG)
        row.pack(fill=tk.X)
        self._license_entry = tk.Entry(
            row, font=self.f_mono, bg=SURFACE, fg=FG,
            insertbackground=FG, relief=tk.FLAT, bd=4, width=28,
        )
        self._license_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self._license_activate_btn = HoverButton(
            row, text="Activate", command=self._activate_license,
            bg=GREEN, fg="#14341b", hover_bg="#7fd46f",
            pady=6, padx=12, font=self.f_ui_bold,
        )
        self._license_activate_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._license_error_lbl = tk.Label(panel, text="", bg=BG, fg=RED, font=self.f_ui)
        self._license_error_lbl.pack(anchor=tk.W, pady=(4, 0))

    def _activate_license(self):
        import threading
        key = (self._license_entry.get() if self._license_entry else "").strip()
        if self._license_error_lbl:
            self._license_error_lbl.config(text="")
        if self._license_activate_btn:
            self._license_activate_btn.set_style("Checking…", bg=SURFACE, fg=FG_MUTE, hover_bg=SURFACE)
        def _run():
            ok, msg = _lic.validate_license(key)
            if ok:
                _lic.save_key(key)
            self.after(0, self._on_license_result, ok, msg)
        threading.Thread(target=_run, daemon=True).start()

    def _on_license_result(self, ok: bool, msg: str):
        if ok:
            self._license_valid = True
            for w in self._cta_container.winfo_children():
                w.destroy()
            self._cta = HoverButton(
                self._cta_container, text="▶  Start Bot", command=self._toggle_bot,
                bg=GREEN, fg="#14341b", hover_bg="#7fd46f",
                pady=12, padx=16, font=self.f_cta,
            )
            self._cta.pack(fill=tk.X, expand=True)
            self._license_entry = None
            self._license_activate_btn = None
            self._license_error_lbl = None
            self._push_log("License activated ✓", "ok")
        else:
            self._push_log(f"License error: {msg}", "err")
            if self._license_entry is not None:
                if self._license_activate_btn:
                    self._license_activate_btn.set_style(
                        "Activate", bg=GREEN, fg="#14341b", hover_bg="#7fd46f"
                    )
                if self._license_error_lbl:
                    self._license_error_lbl.config(text=msg)
            else:
                self._cta.set_style("▶  Start Bot", bg=GREEN, fg="#14341b", hover_bg="#7fd46f")
                self._show_license_panel()
                if self._license_error_lbl:
                    self._license_error_lbl.config(text=msg)

    def _toggle_bot(self):
        if self._bot_thread and self._bot_thread.is_alive():
            self._stop_bot()
        else:
            self._start_bot()

    def _save_settings(self):
        try:
            data = {
                "zone":     self._zone_var.get(),
                "death_sw": self._death_sw.get(),
                "melee_sw": self._melee_sw.get(),
                "rc_sw":    self._rc_sw.get(),
                "rc_cd":    self._rc_cd.get(),
                "k3_sw":    self._k3_sw.get(),
                "k3_cd":    self._k3_cd.get(),
                "mv_sw":    self._mv_sw.get(),
                "mv_cd":    self._mv_cd.get(),
            }
            with open(_SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        try:
            with open(_SETTINGS_FILE) as f:
                data = json.load(f)
            valid_zones = {"Satanic (auto)"} | set(config.SATANIC_FIXED_NODES.keys())
            if data.get("zone") in valid_zones:
                self._zone_var.set(data["zone"])
            for sw_attr, key in (
                ("_death_sw", "death_sw"), ("_melee_sw", "melee_sw"),
                ("_rc_sw", "rc_sw"), ("_k3_sw", "k3_sw"), ("_mv_sw", "mv_sw"),
            ):
                sw = getattr(self, sw_attr, None)
                if sw is not None and key in data:
                    sw.set(bool(data[key]))
            for var_attr, key in (
                ("_rc_cd", "rc_cd"), ("_k3_cd", "k3_cd"), ("_mv_cd", "mv_cd"),
            ):
                var = getattr(self, var_attr, None)
                if var is not None and key in data:
                    var.set(str(data[key]))
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            pass

    def _start_bot(self):
        if not self._license_valid:
            self._push_log("No valid license — activate first", "err")
            return
        import main as _main

        self._stop_event = threading.Event()
        # Open the analyzer log alongside the GUI queue (best-effort).
        try:
            import os as _os
            _log_dir = "debug"
            _os.makedirs(_log_dir, exist_ok=True)
            self._bot_log_file = open(_os.path.join(_log_dir, "bot.log"), "a", encoding="utf-8")
            self._bot_log_file.write(
                f"\n===== Bot session started {__import__('datetime').datetime.now().isoformat(timespec='seconds')} =====\n"
            )
            self._bot_log_file.flush()
        except OSError:
            self._bot_log_file = None
        sys.stdout = _QueueWriter(self._log_queue, log_file=self._bot_log_file)
        sys.stderr = _QueueWriter(self._log_queue, log_file=self._bot_log_file)

        def _cd(sv, default):
            try:
                return max(0.1, float(sv.get()))
            except (ValueError, tk.TclError):
                return default

        _zone_sel = self._zone_var.get()
        kwargs = {
            "stop_event": self._stop_event,
            "enable_death_handling": self._death_sw.get(),
            "enable_mining": self._mining_sw.get(),
            "enable_loot_items": self._loot_sw.get(),
            "enable_debug_snapshots": self._debug_sw.get(),
            "target_zone": None if _zone_sel == "Satanic (auto)" else _zone_sel,
            "enable_rightclick_skill":   self._rc_sw.get(),
            "rightclick_cooldown":       _cd(self._rc_cd, 10.0),
            "enable_key3_skill":         self._k3_sw.get(),
            "key3_cooldown":             _cd(self._k3_cd, 50.0),
            "enable_movement_ability":   self._mv_sw.get(),
            "movement_ability_cooldown": _cd(self._mv_cd, 3.0),
            "melee_mode":                self._melee_sw.get(),
        }
        try:
            sig = inspect.signature(_main.run)
            if "on_vote_reset" in sig.parameters:
                kwargs["on_vote_reset"] = self._on_vote_reset
        except Exception:
            pass

        self._save_settings()
        self._bot_thread = threading.Thread(target=_main.run, kwargs=kwargs, daemon=True)
        self._bot_thread.start()

        self._cta.set_style("■  Stop Bot", bg=RED, fg="#3a1420", hover_bg="#f8a3ba")
        self._status_lbl.config(text="Running", fg=GREEN)
        self._pulse.set(alive=True, color=GREEN)
        self._push_log("Bot started — mode: explore", "info")
        self._start_runtime()
        self._start_stats()
        self.after(500, self._watch_thread)

    def _stop_bot(self):
        if self._stop_event:
            self._stop_event.set()
        self._cta.set_style("Stopping...", bg="#3c3c50", fg=FG_MUTE, hover_bg="#3c3c50")
        self._push_log("Stop requested...", "warn")
        self.after(300, self._check_stopped)

    def _check_stopped(self):
        if self._bot_thread and self._bot_thread.is_alive():
            self.after(200, self._check_stopped)
        else:
            self._on_bot_finished()

    def _watch_thread(self):
        if self._bot_thread and not self._bot_thread.is_alive():
            self._on_bot_finished()
        else:
            self.after(500, self._watch_thread)

    def _on_bot_finished(self):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if getattr(self, "_bot_log_file", None) is not None:
            try:
                self._bot_log_file.close()
            except OSError:
                pass
            self._bot_log_file = None
        self._cta.set_style("▶  Start Bot", bg=GREEN, fg="#14341b", hover_bg="#b8ecb4")
        self._status_lbl.config(text="Idle", fg=FG_MUTE)
        self._pulse.set(alive=False, color=FG_MUTE)
        self._push_log("Bot stopped", "err")
        self._stop_runtime()
        self._tracker.stop()

    def _on_vote_reset(self):
        self.after(0, self._bump_reset)

    def _bump_reset(self):
        self._reset_count += 1
        self._reset_lbl.config(text=str(self._reset_count))
        self._push_log("Vote reset triggered", "reset")

    def _start_runtime(self):
        self._runtime_sec = 0
        self._tick_runtime()

    def _stop_runtime(self):
        if self._runtime_job:
            self.after_cancel(self._runtime_job)
            self._runtime_job = None

    def _tick_runtime(self):
        h = self._runtime_sec // 3600
        m = (self._runtime_sec % 3600) // 60
        s = self._runtime_sec % 60
        self._runtime_lbl.config(text=f"runtime · {h:02d}:{m:02d}:{s:02d}")
        self._runtime_sec += 1
        self._runtime_job = self.after(1000, self._tick_runtime)

    # ── session stats ─────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_gold(n: int) -> str:
        if n >= 1_000_000:
            return f"+{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"+{n:,}"
        return f"+{n}"

    def _on_zone_entered_packet(self, zone_name: str, buff_names: list):
        self.after(0, self._on_zone_entered_main, zone_name, buff_names)

    def _on_zone_entered_main(self, zone_name: str, buff_names: list):
        if self._zone_lbl:
            self._zone_lbl.config(text=zone_name)
        if self._buffs_lbl:
            self._buffs_lbl.config(text=" · ".join(buff_names) if buff_names else "—")
        if self._items_lbl:
            self._items_lbl.config(text="—")
        self._push_log(f"Zone entered: {zone_name}", "info")
        if buff_names:
            self._push_log(f"Buffs: {', '.join(buff_names)}", "info")

    def _on_char_updated_packet(self, char: dict):
        self.after(0, self._on_char_updated_main, char)

    def _on_char_updated_main(self, char: dict):
        if self._char_lbl:
            lv = f"Lv{char['level']}"
            if char.get("herolevel", 0) > 0:
                lv += f" HL{char['herolevel']}"
            self._char_lbl.config(text=f"{char['name']}  ·  {char['class_name']}  ·  {lv}")

    def _on_item_dropped_packet(self, _rarity: int):
        self.after(0, self._on_item_dropped_main)

    def _on_item_dropped_main(self):
        if not self._items_lbl:
            return
        items = self._tracker.items_by_rarity
        if not items:
            self._items_lbl.config(text="—")
            return
        _rarity_short = {1: "M", 2: "R", 3: "H", 4: "A"}
        parts = [
            f"{items[r]}{_rarity_short[r]}"
            for r in sorted(items.keys(), reverse=True)
            if r > 0 and r in _rarity_short
        ]
        self._items_lbl.config(text=" ".join(parts) if parts else "—")

    def _on_kill_packet(self, _exp_delta: int):
        self.after(0, self._on_kill_main)

    def _on_kill_main(self):
        if not self._kills_lbl:
            return
        exp = self._tracker.exp_earned
        if exp <= 0:
            self._kills_lbl.config(text="—")
            return
        if exp >= 1_000_000:
            self._kills_lbl.config(text=f"+{exp / 1_000_000:.2f}M")
        elif exp >= 1_000:
            self._kills_lbl.config(text=f"+{exp / 1_000:.1f}k")
        else:
            self._kills_lbl.config(text=f"+{exp}")

    def _start_stats(self):
        self._stop_stats()
        if self._kills_lbl:
            self._kills_lbl.config(text="—")
        if self._items_lbl:
            self._items_lbl.config(text="—")
        self._tracker.start()
        # _tick_stats reschedules itself, but something has to fire the first
        # one or the gold/rate labels never update at all.
        self._tick_stats()

    def _stop_stats(self):
        if self._stats_job:
            self.after_cancel(self._stats_job)
            self._stats_job = None

    def _tick_stats(self):
        src = self._tracker.gold_source
        if self._gold_earned_lbl and src != "unavailable":
            earned = self._tracker.gold_earned
            per_hr = self._tracker.gold_per_hour
            self._gold_earned_lbl.config(text=self._fmt_gold(earned))
            if self._gold_hr_lbl:
                if per_hr >= 1_000_000:
                    rate_str = f"{per_hr / 1_000_000:.2f}M/hr"
                elif per_hr >= 1_000:
                    rate_str = f"{per_hr / 1_000:.0f}k/hr"
                elif per_hr > 0:
                    rate_str = f"{per_hr}/hr"
                else:
                    rate_str = "—"
                self._gold_hr_lbl.config(text=rate_str)
            if self._gold_src_lbl:
                mode = "live" if (src == "packets" and self._tracker.available) else "save-file"
                self._gold_src_lbl.config(text=f"({mode})")
        elif self._gold_src_lbl and src == "unavailable":
            self._gold_src_lbl.config(text="install Npcap · run as admin for live data")
        self._stats_job = self.after(2000, self._tick_stats)

    def _classify(self, line: str) -> str:
        low = line.lower()
        if "vote reset" in low and "triggered" in low:
            return "reset"
        if any(w in low for w in ("error", "failed", "exception")):
            return "err"
        if any(w in low for w in ("stuck", "warn", "flee", "lockout")):
            return "warn"
        if any(w in low for w in ("kill", "respawn", "ok ")):
            return "ok"
        if any(w in low for w in ("start", "mode:", "zone:", "enemy", "explor")):
            return "info"
        return ""

    def _push_log(self, msg: str, cls: str = ""):
        ts = time.strftime("%H:%M:%S")
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, f"[{ts}] ", "time")
        self._log.insert(tk.END, msg + "\n", cls or self._classify(msg))
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)
        lines = int(self._log.index("end-1c").split(".")[0]) - 1
        self._log_count.config(text=f"{lines} line{'' if lines == 1 else 's'}")

    def _poll_log(self):
        try:
            while True:
                text = self._log_queue.get_nowait()
                for line in text.splitlines():
                    line = line.rstrip()
                    if line:
                        self._push_log(line)
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    def _clear_log(self):
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)
        self._log_count.config(text="0 lines")

    def _on_tk_exception(self, exc, val, tb):
        # Surface UI callback errors inside the log panel instead of failing silently.
        self._push_log("GUI callback error", "err")
        for line in traceback.format_exception(exc, val, tb):
            for part in line.rstrip().splitlines():
                if part:
                    self._push_log(part, "err")

    def _on_close(self):
        if self._bot_thread and self._bot_thread.is_alive():
            if self._stop_event:
                self._stop_event.set()
            self._bot_thread.join(timeout=1.5)
        self._save_settings()
        self._stop_stats()
        self._tracker.stop()
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if getattr(self, "_bot_log_file", None) is not None:
            try:
                self._bot_log_file.close()
            except OSError:
                pass
            self._bot_log_file = None
        self.destroy()


def _ensure_admin() -> bool:
    """
    On Windows, re-launch the current script with admin rights if not already elevated.
    Returns True if already admin (or non-Windows), False if relaunching.
    Only elevates when Npcap is installed — otherwise admin is not needed.
    """
    if sys.platform != "win32":
        return True
    import ctypes
    if ctypes.windll.shell32.IsUserAnAdmin():
        return True
    # Only bother elevating if Npcap is present
    try:
        ctypes.WinDLL("wpcap.dll")
    except OSError:
        return True  # no Npcap — admin not needed
    # Re-launch this script elevated; the UAC dialog will appear
    script = os.path.abspath(__file__)
    script_dir = os.path.dirname(script)
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, f'"{script}"', script_dir, 1
    )
    return False  # current (non-admin) process should exit


def main():
    if not _ensure_admin():
        sys.exit(0)
    app = BotUI()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app._on_close()


if __name__ == "__main__":
    main()
