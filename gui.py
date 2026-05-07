"""
SiegeScouter GUI — native Tkinter (no extra deps)
Drop-in replacement for gui.py. Works on any Python version with tkinter.

Optional: patch main.run() to accept `on_vote_reset=callback` — the GUI
will bump the reset counter each time it fires. If you don't patch, the
counter will stay at 0 (everything else works).
"""

import io
import inspect
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
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str) -> int:
        if text:
            self._q.put(text)
        return len(text)

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

        self.f_ui = tkfont.Font(family="Segoe UI", size=10)
        self.f_ui_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.f_title = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.f_mono = tkfont.Font(family="Consolas", size=9)
        self.f_mono_bold = tkfont.Font(family="Consolas", size=9, weight="bold")
        self.f_cta = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.f_huge = tkfont.Font(family="Segoe UI", size=26, weight="bold")
        self.f_cap = tkfont.Font(family="Consolas", size=8, weight="bold")

        self._build_ui()
        self.report_callback_exception = self._on_tk_exception
        self.after(80, self._poll_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.geometry("460x870")
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
        tk.Label(brand_text, text="by Biggorinho · v0.4", bg=BG, fg=FG_MUTE, font=("Consolas", 8)).pack(anchor=tk.W)

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
        self._mining_sw = _row(opt, "Enable Mining", False, enabled=False, soon=True)
        self._loot_sw = _row(opt, "Enable Loot Items", False, enabled=False, soon=True)

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

    def _start_bot(self):
        if not self._license_valid:
            self._push_log("No valid license — activate first", "err")
            return
        import main as _main

        self._stop_event = threading.Event()
        sys.stdout = _QueueWriter(self._log_queue)
        sys.stderr = _QueueWriter(self._log_queue)

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
            "target_zone": None if _zone_sel == "Satanic (auto)" else _zone_sel,
            "enable_rightclick_skill":   self._rc_sw.get(),
            "rightclick_cooldown":       _cd(self._rc_cd, 10.0),
            "enable_key3_skill":         self._k3_sw.get(),
            "key3_cooldown":             _cd(self._k3_cd, 50.0),
            "enable_movement_ability":   self._mv_sw.get(),
            "movement_ability_cooldown": _cd(self._mv_cd, 3.0),
        }
        try:
            sig = inspect.signature(_main.run)
            if "on_vote_reset" in sig.parameters:
                kwargs["on_vote_reset"] = self._on_vote_reset
        except Exception:
            pass

        self._bot_thread = threading.Thread(target=_main.run, kwargs=kwargs, daemon=True)
        self._bot_thread.start()

        self._cta.set_style("■  Stop Bot", bg=RED, fg="#3a1420", hover_bg="#f8a3ba")
        self._status_lbl.config(text="Running", fg=GREEN)
        self._pulse.set(alive=True, color=GREEN)
        self._push_log("Bot started — mode: explore", "info")
        self._start_runtime()
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
        self._cta.set_style("▶  Start Bot", bg=GREEN, fg="#14341b", hover_bg="#b8ecb4")
        self._status_lbl.config(text="Idle", fg=FG_MUTE)
        self._pulse.set(alive=False, color=FG_MUTE)
        self._push_log("Bot stopped", "err")
        self._stop_runtime()

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
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self.destroy()


def main():
    app = BotUI()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app._on_close()


if __name__ == "__main__":
    main()
