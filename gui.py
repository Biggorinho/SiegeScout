import sys
import threading
import tkinter as tk
from tkinter import scrolledtext
import io
import queue

import config


# ── stdout redirect ────────────────────────────────────────────────────────────

class _QueueWriter(io.TextIOBase):
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str) -> int:
        if text:
            self._q.put(text)
        return len(text)

    def flush(self):
        pass


# ── main GUI ───────────────────────────────────────────────────────────────────

class BotUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hero Siege Bot")
        self.resizable(False, False)
        self.configure(bg="#1e1e2e")

        self._stop_event: threading.Event | None = None
        self._bot_thread: threading.Thread | None = None
        self._log_queue: queue.Queue = queue.Queue()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        self._build_ui()
        self.after(100, self._poll_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = dict(padx=10, pady=6)
        BG  = "#1e1e2e"
        FG  = "#cdd6f4"
        ACC = "#89b4fa"
        SEP = "#313244"

        # ── header ──
        hdr = tk.Frame(self, bg="#181825", pady=6)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Hero Siege Bot", font=("Segoe UI", 14, "bold"),
                 bg="#181825", fg=ACC).pack()

        # ── status row ──
        status_row = tk.Frame(self, bg=BG)
        status_row.pack(fill=tk.X, **PAD)
        tk.Label(status_row, text="Status:", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Idle")
        self._status_lbl = tk.Label(status_row, textvariable=self._status_var,
                                    bg=BG, fg="#a6e3a1", font=("Segoe UI", 10, "bold"))
        self._status_lbl.pack(side=tk.LEFT, padx=6)

        # ── start/stop button ──
        self._start_btn = tk.Button(self, text="▶  Start Bot",
                                    command=self._toggle_bot,
                                    font=("Segoe UI", 11, "bold"),
                                    bg="#a6e3a1", fg="#1e1e2e",
                                    activebackground="#94d3a0",
                                    relief=tk.FLAT, cursor="hand2",
                                    padx=20, pady=6)
        self._start_btn.pack(padx=10, pady=(0, 6))

        # ── separator ──
        tk.Frame(self, bg=SEP, height=1).pack(fill=tk.X, padx=10)

        # ── options ──
        opt_frame = tk.LabelFrame(self, text="Options", bg=BG, fg=ACC,
                                  font=("Segoe UI", 10, "bold"),
                                  bd=1, relief=tk.GROOVE)
        opt_frame.pack(fill=tk.X, padx=10, pady=8)

        def _chk(parent, label, default, enabled=True):
            var = tk.BooleanVar(value=default)
            tk.Checkbutton(parent, text=label, variable=var,
                           bg=BG, fg=FG, selectcolor=SEP,
                           activebackground=BG, activeforeground=FG,
                           font=("Segoe UI", 10),
                           state=(tk.NORMAL if enabled else tk.DISABLED),
                           disabledforeground="#7f849c").pack(anchor=tk.W, padx=8, pady=1)
            return var

        self._death_var   = _chk(opt_frame, "Enable Death Handling",  config.ENABLE_DEATH_HANDLING)
        self._mining_var  = _chk(opt_frame, "Enable Mining (Coming soon)", False, enabled=False)
        self._loot_items_var = _chk(opt_frame, "Enable Loot Items (Coming soon)", False, enabled=False)

        # ── separator ──
        tk.Frame(self, bg=SEP, height=1).pack(fill=tk.X, padx=10)

        # ── log area ──
        log_frame = tk.LabelFrame(self, text="Log", bg=BG, fg=ACC,
                                  font=("Segoe UI", 10, "bold"),
                                  bd=1, relief=tk.GROOVE)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self._log = scrolledtext.ScrolledText(
            log_frame, height=14, width=62,
            bg="#11111b", fg="#cdd6f4",
            font=("Consolas", 9),
            state=tk.DISABLED, relief=tk.FLAT,
            insertbackground=FG,
        )
        self._log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        clear_btn = tk.Button(log_frame, text="Clear Log",
                              command=self._clear_log,
                              font=("Segoe UI", 9),
                              bg=SEP, fg=FG, relief=tk.FLAT,
                              cursor="hand2", padx=8, pady=2)
        clear_btn.pack(anchor=tk.E, padx=4, pady=(0, 4))

        self.geometry("560x540")

    # ── bot lifecycle ──────────────────────────────────────────────────────────

    def _toggle_bot(self):
        if self._bot_thread and self._bot_thread.is_alive():
            self._stop_bot()
        else:
            self._start_bot()

    def _start_bot(self):
        import main as _main

        self._stop_event = threading.Event()
        sys.stdout = _QueueWriter(self._log_queue)
        sys.stderr = _QueueWriter(self._log_queue)

        kwargs = dict(
            stop_event=self._stop_event,
            enable_death_handling=self._death_var.get(),
            enable_mining=self._mining_var.get(),
            enable_loot_items=self._loot_items_var.get(),
        )
        self._bot_thread = threading.Thread(target=_main.run, kwargs=kwargs, daemon=True)
        self._bot_thread.start()

        self._start_btn.config(text="■  Stop Bot", bg="#f38ba8", activebackground="#d37898")
        self._status_var.set("Running")
        self._status_lbl.config(fg="#f38ba8")
        self.after(500, self._watch_thread)

    def _stop_bot(self):
        if self._stop_event:
            self._stop_event.set()
        self._start_btn.config(state=tk.DISABLED, text="Stopping…")
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
        self._start_btn.config(text="▶  Start Bot", bg="#a6e3a1",
                               activebackground="#94d3a0", state=tk.NORMAL)
        self._status_var.set("Idle")
        self._status_lbl.config(fg="#a6e3a1")

    # ── log polling ────────────────────────────────────────────────────────────

    def _poll_log(self):
        try:
            while True:
                text = self._log_queue.get_nowait()
                self._log.config(state=tk.NORMAL)
                self._log.insert(tk.END, text)
                self._log.see(tk.END)
                self._log.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    def _clear_log(self):
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

    # ── close ──────────────────────────────────────────────────────────────────

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
        # Allow clean terminal stop (Ctrl+C) without traceback noise.
        app._on_close()


if __name__ == "__main__":
    main()
