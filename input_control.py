import ctypes
import ctypes.wintypes
import math
import random
import time

import keyboard

from config import (
    SCREEN_W, SCREEN_H,
    MOVE_AXIS_DEADZONE,
    STUCK_BACKTRACK_DURATION_MIN, STUCK_BACKTRACK_DURATION_MAX, STUCK_BACKTRACK_COOLDOWN,
    JUMP_COOLDOWN, JUMP_ALIGNMENT_THRESHOLD, JUMP_AIM_DISTANCE,
    AIM_MIN_DISTANCE, AIM_MAX_DISTANCE,
    KEY3_PRESS_HOLD_SECONDS, KEY3_REPEAT_COUNT, KEY3_DOUBLE_TAP_GAP_SECONDS,
    ATTACK_SWEEP_DURATION_SECONDS, ATTACK_SWEEP_RADIUS,
)


def sleep_interruptible(duration, step=0.02):
    """Sleep in short slices; return False if user pressed F1 or p."""
    end_t = time.time() + max(0.0, duration)
    while time.time() < end_t:
        if keyboard.is_pressed('f1') or keyboard.is_pressed('f8') or keyboard.is_pressed('p'):
            return False
        time.sleep(min(step, max(0.0, end_t - time.time())))
    return True


def left_mouse_down():
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)


def left_mouse_up():
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)


def attack_key_down():
    """Press Q (the attack key)."""
    keyboard.press('q')


def attack_key_up():
    """Release Q (the attack key)."""
    keyboard.release('q')


def right_mouse_click(hold_time=0.05):
    ctypes.windll.user32.mouse_event(0x0008, 0, 0, 0, 0)
    sleep_interruptible(hold_time)
    ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0)


def left_click_at(x, y, hold_time=0.05):
    ctypes.windll.user32.SetCursorPos(int(x), int(y))
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
    sleep_interruptible(hold_time)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)


def tap_key(key_name, hold_time=0.05):
    keyboard.press(key_name)
    sleep_interruptible(hold_time)
    keyboard.release(key_name)


def tap_key_shifted(key_name, hold_time=0.05):
    keyboard.press('shift')
    keyboard.press(key_name)
    sleep_interruptible(hold_time)
    keyboard.release(key_name)
    keyboard.release('shift')


def tap_key_uninterruptible(key_name, hold_time=0.05):
    """Tap key without q/p interrupt checks for scripted short sequences."""
    keyboard.press(key_name)
    time.sleep(max(0.0, hold_time))
    keyboard.release(key_name)


def tap_key_vk_uninterruptible(vk_code, hold_time=0.05):
    """Tap a key by virtual-key code using the Windows keyboard API."""
    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
    time.sleep(max(0.0, hold_time))
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)


def tap_key_sendinput_uninterruptible(vk_code, hold_time=0.05):
    """Tap a key using SendInput with scan codes for better reliability."""
    user32           = ctypes.windll.user32
    KEYEVENTF_KEYUP  = 0x0002
    KEYEVENTF_SCANCODE = 0x0008

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk",         ctypes.wintypes.WORD),
            ("wScan",       ctypes.wintypes.WORD),
            ("dwFlags",     ctypes.wintypes.DWORD),
            ("time",        ctypes.wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class INPUTUNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.wintypes.DWORD), ("union", INPUTUNION)]

    scan_code = user32.MapVirtualKeyW(vk_code, 0)
    key_down  = INPUT(type=1, union=INPUTUNION(ki=KEYBDINPUT(0, scan_code, KEYEVENTF_SCANCODE, 0, 0)))
    key_up    = INPUT(type=1, union=INPUTUNION(ki=KEYBDINPUT(0, scan_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 0)))
    user32.SendInput(1, ctypes.byref(key_down), ctypes.sizeof(INPUT))
    time.sleep(max(0.0, hold_time))
    user32.SendInput(1, ctypes.byref(key_up), ctypes.sizeof(INPUT))


def tap_space(hold_time=0.04):
    keyboard.press('space')
    sleep_interruptible(hold_time)
    keyboard.release('space')


def release_all_inputs():
    """Emergency input release for movement keys and held attack inputs."""
    for k in ('w', 'a', 's', 'd'):
        keyboard.release(k)
    # Attack is bound to Q, but release LMB too as a safety net.
    attack_key_up()
    left_mouse_up()


def wasd_move(dx, dy, duration=0.9):
    """Press WASD keys in the direction of (dx, dy) for `duration` seconds."""
    keys = _direction_keys(dx, dy)
    try:
        for k in keys:
            keyboard.press(k)
        sleep_interruptible(duration)
    finally:
        for k in keys:
            keyboard.release(k)


def explore_move_start(dx, dy, held_keys=frozenset()):
    """Hold WASD keys for (dx, dy) without releasing between ticks.
    Releases only the keys that changed, presses new ones.
    Returns the new frozenset of held keys."""
    new_keys = frozenset(_direction_keys(dx, dy))
    for k in held_keys - new_keys:
        keyboard.release(k)
    for k in new_keys - held_keys:
        keyboard.press(k)
    return new_keys


def explore_move_stop(held_keys=frozenset()):
    """Release all held WASD explore keys. Returns empty frozenset."""
    for k in held_keys:
        keyboard.release(k)
    return frozenset()


def backtrack_move(dx, dy, last_backtrack_t):
    """Reverse briefly to escape a corner. `last_backtrack_t` is a one-element mutable list."""
    now = time.time()
    if now - last_backtrack_t[0] < STUCK_BACKTRACK_COOLDOWN:
        return
    wasd_move(-dx, -dy, random.uniform(STUCK_BACKTRACK_DURATION_MIN, STUCK_BACKTRACK_DURATION_MAX))
    last_backtrack_t[0] = now


def cast_right_click_skill(attack_held, resume_left=False):
    """Cast right-click skill while managing held Q attack."""
    if attack_held:
        attack_key_up()
        attack_held = False
        time.sleep(0.15)
    right_mouse_click(hold_time=0.06)
    if resume_left:
        time.sleep(0.05)
        attack_key_down()
        attack_held = True
    return attack_held


def cast_key_skill_safe(key_name, attack_held, resume_left=False, hold_time=0.06):
    """Cast a keyboard skill safely: avoid overlap with held attack key."""
    if attack_held:
        attack_key_up()
        attack_held = False
        sleep_interruptible(0.02)
    tap_key(key_name, hold_time=hold_time)
    if resume_left:
        attack_key_down()
        attack_held = True
        sleep_interruptible(0.02)
    return attack_held


def cast_key3_skill_safe(attack_held, resume_left=False):
    """Cast key 3 reliably: release attack, double tap, then resume attack."""
    if attack_held:
        attack_key_up()
        attack_held = False
        sleep_interruptible(0.03)
    for i in range(KEY3_REPEAT_COUNT):
        tap_key_uninterruptible('3', hold_time=KEY3_PRESS_HOLD_SECONDS)
        if i < KEY3_REPEAT_COUNT - 1:
            time.sleep(KEY3_DOUBLE_TAP_GAP_SECONDS)
    if resume_left:
        attack_key_down()
        attack_held = True
        sleep_interruptible(0.03)
    return attack_held


def spin_to_clear_melee(rounds=2, radius=200, steps=36):
    """Hold Q key and sweep cursor in circles to kill nearby melee enemies."""
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    keyboard.press('q')
    for _ in range(rounds):
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            ctypes.windll.user32.SetCursorPos(
                int(cx + math.cos(angle) * radius),
                int(cy + math.sin(angle) * radius),
            )
            time.sleep(0.04)
    keyboard.release('q')
    ctypes.windll.user32.SetCursorPos(cx, cy)
    time.sleep(0.20)


def cast_opening_buff_sequence(move_dx=0, move_dy=0):
    """Zone opening: walk off spawn, cast shield, then moving spin + x/q/x."""
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    move_keys = _direction_keys(move_dx, move_dy)

    release_all_inputs()

    # Walk first to leave any spawn waypoint under the player before mouse clicks.
    for k in move_keys:
        keyboard.press(k)
    try:
        time.sleep(0.70)

        # Shield must be up before the opening spin.
        if not _aim_cursor_toward_vector(move_dx, move_dy, distance=260):
            ctypes.windll.user32.SetCursorPos(cx + 220, cy)
        right_mouse_click(hold_time=0.10)
        time.sleep(0.20)
        if not _aim_cursor_toward_vector(move_dx, move_dy, distance=260):
            ctypes.windll.user32.SetCursorPos(cx + 220, cy)
        right_mouse_click(hold_time=0.10)
        time.sleep(0.18)

        spin_radius, spin_steps, spin_rounds = 180, 48, 2
        keyboard.press('q')
        for _ in range(spin_rounds):
            for i in range(spin_steps):
                angle = 2 * math.pi * i / spin_steps
                ctypes.windll.user32.SetCursorPos(
                    int(cx + math.cos(angle) * spin_radius),
                    int(cy + math.sin(angle) * spin_radius),
                )
                time.sleep(0.06)
        keyboard.release('q')

        ctypes.windll.user32.SetCursorPos(cx, cy)
        time.sleep(0.20)
    finally:
        for k in move_keys:
            keyboard.release(k)


def _aim_cursor_toward_vector(move_dx, move_dy, distance=JUMP_AIM_DISTANCE):
    move_len = math.hypot(move_dx, move_dy)
    if move_len < 1e-6:
        return False
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    tx = int(max(50, min(SCREEN_W - 50, cx + (move_dx / move_len) * distance)))
    ty = int(max(50, min(SCREEN_H - 50, cy + (move_dy / move_len) * distance)))
    ctypes.windll.user32.SetCursorPos(tx, ty)
    return True


def maybe_jump_with_movement(move_dx, move_dy, cooldown_tracker, force_aim=False):
    """Tap jump when movement and cursor align; if force_aim, point cursor first."""
    now = time.time()
    if now - cooldown_tracker[0] < JUMP_COOLDOWN:
        return

    move_len = math.hypot(move_dx, move_dy)
    if move_len < 1e-6:
        return

    if force_aim:
        try:
            if _aim_cursor_toward_vector(move_dx, move_dy):
                tap_space()
                cooldown_tracker[0] = now
        except Exception:
            pass
        return

    try:
        pt = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    except Exception:
        return

    cursor_dx  = pt.x - (SCREEN_W // 2)
    cursor_dy  = pt.y - (SCREEN_H // 2)
    cursor_len = math.hypot(cursor_dx, cursor_dy)
    if cursor_len < 1e-6:
        return

    alignment = (move_dx * cursor_dx + move_dy * cursor_dy) / (move_len * cursor_len)
    if alignment >= JUMP_ALIGNMENT_THRESHOLD:
        tap_space()
        cooldown_tracker[0] = now


def maybe_jump_stuck_sequence(
    preferred_dx,
    preferred_dy,
    cooldown_tracker,
    sequence_index_tracker,
    preferred_pending_tracker,
):
    """Stuck jump policy: try blocked-wall direction once, then cycle L/R/U/D."""
    now = time.time()
    if now - cooldown_tracker[0] < JUMP_COOLDOWN:
        return False, None

    if preferred_pending_tracker[0]:
        preferred_len = math.hypot(preferred_dx, preferred_dy)
        if preferred_len > 1e-6:
            try:
                if _aim_cursor_toward_vector(preferred_dx, preferred_dy):
                    tap_space()
                    cooldown_tracker[0] = now
                    preferred_pending_tracker[0] = False
                    return True, "wall"
            except Exception:
                pass
        preferred_pending_tracker[0] = False

    sequence = [
        (-1.0, 0.0, "left"),
        (1.0, 0.0, "right"),
        (0.0, -1.0, "up"),
        (0.0, 1.0, "down"),
    ]
    idx = int(sequence_index_tracker[0]) % len(sequence)
    jump_dx, jump_dy, label = sequence[idx]

    try:
        if _aim_cursor_toward_vector(jump_dx, jump_dy):
            tap_space()
            cooldown_tracker[0] = now
            sequence_index_tracker[0] = (idx + 1) % len(sequence)
            return True, label
    except Exception:
        pass
    return False, None


def attack_toward(dx, dy):
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    length = math.hypot(dx, dy)
    if length < 1e-6:
        tx, ty = cx, cy
    else:
        scale = random.uniform(AIM_MIN_DISTANCE, AIM_MAX_DISTANCE) / length
        tx    = int(cx + dx * scale + random.randint(-10, 10))
        ty    = int(cy + dy * scale + random.randint(-10, 10))
    tx = max(50, min(SCREEN_W - 50, tx))
    ty = max(50, min(SCREEN_H - 50, ty))
    ctypes.windll.user32.SetCursorPos(tx, ty)


def wasd_move_and_attack(move_dx, move_dy, duration=0.32, aim_dx=None, aim_dy=None):
    """Move with WASD while left-click is held for continuous auto-fire."""
    if aim_dx is None:
        aim_dx = move_dx
    if aim_dy is None:
        aim_dy = move_dy
    keys = _direction_keys(move_dx, move_dy)
    try:
        for k in keys:
            keyboard.press(k)
        end_t = time.time() + duration
        while time.time() < end_t:
            if keyboard.is_pressed('f1') or keyboard.is_pressed('p'):
                break
            keyboard.press('q')
            attack_toward(aim_dx, aim_dy)
            if not sleep_interruptible(0.02):
                break
    finally:
        for k in keys:
            keyboard.release(k)


def perform_attack_sweep(
    duration=ATTACK_SWEEP_DURATION_SECONDS,
    radius=ATTACK_SWEEP_RADIUS,
    move_dx=0,
    move_dy=0,
):
    """Sweep cursor around player while optionally moving with WASD."""
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    points = [
        (0.0, -1.0), (0.7, -0.7), (1.0, 0.0), (0.7, 0.7), (0.0, 1.0),
        (-0.7, 0.7), (-1.0, 0.0), (-0.7, -0.7), (0.0, -1.0),
    ]
    seg_dur   = max(0.08, duration / max(1, len(points) - 1))
    move_keys = _direction_keys(move_dx, move_dy)
    try:
        for k in move_keys:
            keyboard.press(k)
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            seg_start = time.time()
            while (time.time() - seg_start) < seg_dur:
                if keyboard.is_pressed('f1') or keyboard.is_pressed('p'):
                    return
                t  = min(1.0, (time.time() - seg_start) / seg_dur)
                vx = x1 * (1.0 - t) + x2 * t
                vy = y1 * (1.0 - t) + y2 * t
                tx = int(max(50, min(SCREEN_W - 50, cx + vx * radius)))
                ty = int(max(50, min(SCREEN_H - 50, cy + vy * radius)))
                ctypes.windll.user32.SetCursorPos(tx, ty)
                keyboard.press('q')
                if not sleep_interruptible(0.035):
                    return
    finally:
        for k in move_keys:
            keyboard.release(k)


def _direction_keys(dx, dy):
    """Return the WASD keys needed for movement vector (dx, dy)."""
    keys = []
    if dy < -MOVE_AXIS_DEADZONE:
        keys.append('w')
    if dy > MOVE_AXIS_DEADZONE:
        keys.append('s')
    if dx < -MOVE_AXIS_DEADZONE:
        keys.append('a')
    if dx > MOVE_AXIS_DEADZONE:
        keys.append('d')
    return keys or ['d']
