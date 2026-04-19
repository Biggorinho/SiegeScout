import ctypes
import ctypes.wintypes
import cv2
import mss
import numpy as np
import os
import random
import time
import math
import keyboard

from config import (
    SCREEN_W, SCREEN_H, DEBUG_DIR,
    MINIMAP_MODE,
    HP_FLEE_THRESHOLD, HP_RESUME_THRESHOLD,
    HP_ZERO_DEAD_THRESHOLD, HP_ZERO_RESPAWN_COOLDOWN_SECONDS,
    HP_BAR_MISSING_GRACE_SECONDS,
    HEALTH_POTION_KEY, HEALTH_POTION_COOLDOWN_SECONDS,
    HEALTH_POTION_BURST_MIN_HP, HEALTH_POTION_BURST_MAX_HP,
    HEALTH_POTION_BURST_COUNT, HEALTH_POTION_BURST_GAP_SECONDS,
    ATTACK_LOST_GRACE, ATTACK_TIMEOUT, LOOT_TIMEOUT, FLEE_MAX_WAIT,
    ATTACK_MOVE_REPLAN_SECONDS,
    ATTACK_RIGHTCLICK_INTERVAL_SECONDS, ATTACK_KEY3_INTERVAL_SECONDS,
    ATTACK_SWEEP_INTERVAL_SECONDS, ATTACK_SWEEP_DURATION_SECONDS, ATTACK_SWEEP_RADIUS,
    ATTACK_NO_CONFIRMED_ENEMY_TIMEOUT_SECONDS,
    ATTACK_STUCK_WINDOW, ATTACK_STUCK_MIN_MOVE,
    ATTACK_STUCK_ARM_SECONDS,
    EXPLORE_REPLAN_SECONDS, EXPLORE_MOVE_MIN_SECONDS, EXPLORE_MOVE_MAX_SECONDS,
    POST_KILL_SCAN_MIN_SECONDS, POST_KILL_SCAN_MAX_SECONDS,
    STUCK_LOG_COOLDOWN, STUCK_COUNT_EVENT_COOLDOWN_SECONDS,
    STUCK_VOTE_RESET_CONSECUTIVE, HARD_STUCK_RESET_SECONDS,
    HARD_STUCK_SNAPSHOT_COOLDOWN_SECONDS,
    CORNER_ESCAPE_COOLDOWN_SECONDS, CORNER_ESCAPE_MAX_ATTEMPTS_BEFORE_RESET,
    VOTE_RESET_COOLDOWN_SECONDS, VOTE_RESET_ENEMY_GRACE_SECONDS,
    VOTE_RESET_MIN_HP, NO_ENEMY_RESET_MIN_FRAMES,
    ZONE_MAX_SECONDS_BEFORE_RESET, ZONE_MIN_EXPLORE_BEFORE_RESET_SECONDS,
    ZONE_TRANSITION_FREEZE_SECONDS,
    ZONE_OPENING_KEY3_START_TAPS,
    DEATH_RESPAWN_COOLDOWN_SECONDS, DEATH_CONFIRM_FRAMES, ENABLE_DEATH_HANDLING,
    RESPAWN_TO_RESET_DELAY_SECONDS,
    ENEMY_CONFIRM_FRAMES, ENEMY_LOST_FRAMES,
    LAST_SEEN_MONSTER_MEMORY_SECONDS,
    COMPANION_TRACK_RADIUS,
    RESET_DIAG_PRINT_INTERVAL_SECONDS,
    ATTACK_IDLE_VOTE_RESET_SECONDS,
    ITEM_TEMPLATE_PATHS, ENABLE_LOOT,
)
from detectors import (
    classify_enemy_dots, get_hp_percent, find_items_on_screen,
    is_death_screen, is_world_map_visible,
)
from minimap import (
    get_minimap_region, minimap_center_from_image,
    dot_distance, dot_to_screen_delta,
    build_walkable_minimap_mask, buffered_walkable_mask,
    random_explore_delta,
)
from pathfinding import (
    ray_walkability_score, choose_explore_delta, choose_recent_enemy_delta,
    choose_attack_move_delta, _a_star_next_waypoint,
)
from state import State, StuckDetector, ExploreHeatMap, TerrainMemory, CompanionSuppressor
from input_control import (
    sleep_interruptible, wasd_move, backtrack_move,
    left_mouse_down, left_mouse_up,
    cast_right_click_skill, cast_key3_skill_safe,
    spin_to_clear_melee, cast_opening_buff_sequence,
    tap_key, release_all_inputs,
    _aim_cursor_toward_vector, maybe_jump_with_movement,
    wasd_move_and_attack, perform_attack_sweep,
)
from combat import (
    run_vote_reset_sequence, attempt_respawn, perform_corner_escape_burst,
    pick_zone_from_open_map,
)
from debug_utils import save_hard_stuck_snapshot, save_minimap_frame


def run():
    item_templates = []
    for path in ITEM_TEMPLATE_PATHS:
        if os.path.exists(path):
            item_templates.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            print(f"Template: {path}")
        else:
            item_templates.append(None)
            print(f"Fehlt: {path}")

    print("Bot gestartet | q=Stop | p=Pause | esc=Emergency stop | f6=Minimap Mode")
    print("Spiel muss sichtbar sein (kein Hintergrundmodus mit mss)")
    print(f"Minimap mode: {MINIMAP_MODE}")

    state              = State.EXPLORE
    paused             = False
    stuck              = StuckDetector()
    state_start        = time.time()
    last_flee_dir      = (0, 0)
    last_enemy_seen_t  = 0.0
    adx, ady           = 0, 0
    explore_dx, explore_dy       = random_explore_delta()
    last_explore_plan_t          = 0.0
    recent_explore_angles        = []
    blocked_explore_angles       = []
    attack_held                  = False
    last_stuck_log_t             = 0.0
    attack_strafe_sign           = 1
    last_attack_move_plan_t      = 0.0
    attack_move_dx, attack_move_dy = random_explore_delta()
    explore_pause_until          = 0.0
    last_jump_t                  = [0.0]
    kite_committed_until         = 0.0
    kite_cached_move             = None
    last_backtrack_t             = [0.0]
    explore_heatmap              = ExploreHeatMap()
    terrain_memory               = TerrainMemory()
    attack_stuck                 = StuckDetector(window=ATTACK_STUCK_WINDOW, min_move=ATTACK_STUCK_MIN_MOVE)
    cached_mask                  = None
    companion                    = CompanionSuppressor()
    minimap_mode                 = MINIMAP_MODE
    debug_mode                   = False
    debug_frame_idx              = 0
    last_attack_activity_t       = time.time()
    zone_had_attack              = False
    last_vote_reset_t            = 0.0
    auto_esc_ignore_until        = 0.0
    auto_q_ignore_until          = 0.0
    zone_transition_until        = 0.0
    last_health_potion_t         = 0.0
    last_known_hp                = None
    next_attack_rightclick_t     = time.time() + ATTACK_RIGHTCLICK_INTERVAL_SECONDS
    next_attack_key3_t           = time.time() + ATTACK_KEY3_INTERVAL_SECONDS
    last_attack_sweep_t          = time.time()
    hp_zero_since                = 0.0
    hp_bar_missing_until         = 0.0
    stuck_no_enemy_count         = 0
    last_stuck_count_t           = 0.0
    no_enemy_frames              = 0
    enemy_visible_frames         = 0
    enemy_lost_frames            = 0
    hard_stuck_since             = 0.0
    corner_escape_attempts       = 0
    last_corner_escape_t         = 0.0
    last_hard_stuck_snapshot_t   = 0.0
    last_seen_enemy_dot          = None
    last_seen_enemy_vec          = None
    last_seen_enemy_t            = 0.0
    last_reset_diag_t            = 0.0
    zone_started_t               = time.time()
    last_death_respawn_t         = 0.0
    death_screen_frames          = 0
    zone_opening_rightclick_pending  = True
    zone_opening_key3_taps_remaining = 0
    next_opening_key3_t          = 0.0
    rightclick_busy_until        = 0.0
    last_explore_target          = None

    # ── shared reset helper ────────────────────────────────────────────────────
    # Defined as a closure so it can assign back into run()'s local scope.
    def _do_zone_reset():
        nonlocal last_vote_reset_t, zone_transition_until, last_attack_activity_t
        nonlocal zone_had_attack, stuck_no_enemy_count, hard_stuck_since
        nonlocal corner_escape_attempts, no_enemy_frames, hp_bar_missing_until
        nonlocal zone_started_t, zone_opening_rightclick_pending, state, state_start
        nonlocal explore_pause_until, death_screen_frames, hp_zero_since
        nonlocal last_explore_plan_t, explore_dx, explore_dy, last_explore_target
        last_vote_reset_t             = time.time()
        zone_transition_until         = last_vote_reset_t + ZONE_TRANSITION_FREEZE_SECONDS
        last_attack_activity_t        = zone_transition_until
        zone_had_attack               = False
        stuck_no_enemy_count          = 0
        hard_stuck_since              = 0.0
        corner_escape_attempts        = 0
        no_enemy_frames               = 0
        hp_bar_missing_until          = zone_transition_until + HP_BAR_MISSING_GRACE_SECONDS
        zone_started_t                = zone_transition_until
        zone_opening_rightclick_pending = True
        state                         = State.EXPLORE
        state_start                   = last_vote_reset_t
        explore_pause_until           = zone_transition_until
        death_screen_frames           = 0
        hp_zero_since                 = 0.0
        last_explore_plan_t           = 0.0
        explore_dx, explore_dy        = random_explore_delta()
        last_explore_target           = None

        # New zone means new topology; clear cross-zone memory to avoid stale bias.
        recent_explore_angles.clear()
        blocked_explore_angles.clear()
        explore_heatmap.reset()
        terrain_memory.reset()

    with mss.mss() as sct:
        full_mon = {"top": 0, "left": 0, "width": SCREEN_W, "height": SCREEN_H}

        # ── startup: pick satanic zone from the world map ─────────────────────
        startup_frame = cv2.cvtColor(np.array(sct.grab(full_mon)), cv2.COLOR_BGRA2BGR)
        if is_world_map_visible(startup_frame):
            print("[STARTUP] Karte bereits offen – waehle satanische Zone ...")
            if pick_zone_from_open_map(sct, full_mon, startup_frame):
                _do_zone_reset()
        else:
            print("[STARTUP] Oeffne Karte und waehle satanische Zone ...")
            auto_esc_ignore_until = time.time() + 1.0
            if run_vote_reset_sequence(sct, full_mon):
                _do_zone_reset()

        while True:
            now_loop = time.time()

            # ── hotkeys ───────────────────────────────────────────────────────
            if keyboard.is_pressed('esc') and now_loop > auto_esc_ignore_until:
                release_all_inputs()
                attack_held = False
                print("EMERGENCY STOP (esc)")
                break
            if keyboard.is_pressed('q') and now_loop > auto_q_ignore_until:
                if attack_held:
                    left_mouse_up()
                    attack_held = False
                print("Bot gestoppt.")
                break
            if keyboard.is_pressed('p'):
                paused = not paused
                if paused and attack_held:
                    left_mouse_up()
                    attack_held = False
                print("PAUSE" if paused else "RESUME")
                time.sleep(0.5)
                continue
            if keyboard.is_pressed('f6'):
                minimap_mode = "large" if minimap_mode == "small" else "small"
                print(f"Minimap mode -> {minimap_mode}")
                time.sleep(0.35)
                continue
            if keyboard.is_pressed('f7'):
                debug_mode = not debug_mode
                print(f"DEBUG {'AN' if debug_mode else 'AUS'} | frames -> {DEBUG_DIR}/")
                if debug_mode:
                    os.makedirs(DEBUG_DIR, exist_ok=True)
                time.sleep(0.35)
                continue
            if paused:
                time.sleep(0.1)
                continue

            # ── zone transition freeze ────────────────────────────────────────
            if now_loop < zone_transition_until:
                release_all_inputs()
                attack_held = False
                sleep_interruptible(0.08)
                continue

            frame = cv2.cvtColor(np.array(sct.grab(full_mon)), cv2.COLOR_BGRA2BGR)

            # Read HP early so the death-screen gate uses the current frame.
            hp = get_hp_percent(frame)
            if hp is not None:
                last_known_hp = hp

            # ── death screen ──────────────────────────────────────────────────
            # Allow death-screen check when: HP bar is gone (hp is None — dead or
            # transitioning) OR HP bar is visible but critically low.
            # This catches both slow deaths and instant kills without false-triggering
            # during normal combat where the bar is clearly visible and healthy.
            _hp_critical = (hp is None or hp <= HP_ZERO_DEAD_THRESHOLD * 5)
            if ENABLE_DEATH_HANDLING and now_loop >= hp_bar_missing_until and _hp_critical:
                if is_death_screen(frame):
                    death_screen_frames += 1
                else:
                    death_screen_frames = 0

                if death_screen_frames >= DEATH_CONFIRM_FRAMES and \
                        (now_loop - last_death_respawn_t) >= DEATH_RESPAWN_COOLDOWN_SECONDS:
                    release_all_inputs()
                    attack_held = False
                    print("[DEATH] Respawn + Vote Reset")
                    last_death_respawn_t = now_loop
                    attempt_respawn(sct, full_mon)
                    sleep_interruptible(RESPAWN_TO_RESET_DELAY_SECONDS)
                    post_respawn = cv2.cvtColor(np.array(sct.grab(full_mon)), cv2.COLOR_BGRA2BGR)
                    if is_world_map_visible(post_respawn):
                        tap_key('esc', hold_time=0.10)
                        sleep_interruptible(0.40)
                    auto_esc_ignore_until = time.time() + 1.0
                    if run_vote_reset_sequence(sct, full_mon):
                        _do_zone_reset()
                    death_screen_frames = 0
                    continue

            # ── zone opening buff sequence ────────────────────────────────────
            if zone_opening_rightclick_pending and now_loop >= zone_transition_until:
                release_all_inputs()
                attack_held = False
                auto_q_ignore_until = time.time() + 0.8

                # Precompute an opening move from minimap pathfinding so the spin
                # starts into a walkable lane instead of a random corner direction.
                opening_dx, opening_dy = explore_dx, explore_dy
                opening_region = get_minimap_region(minimap_mode)
                opening_minimap = cv2.cvtColor(np.array(sct.grab(opening_region)), cv2.COLOR_BGRA2BGR)
                opening_center = minimap_center_from_image(opening_minimap)
                terrain_memory.update(opening_minimap)
                (opening_move, opening_target) = choose_explore_delta(
                    opening_minimap,
                    opening_center,
                    recent_explore_angles,
                    blocked_explore_angles,
                    heatmap=explore_heatmap,
                    terrain_memory=terrain_memory,
                )
                if opening_move is not None:
                    opening_dx, opening_dy = opening_move
                    explore_dx, explore_dy = opening_move
                if opening_target is not None:
                    angle = math.atan2(opening_target[1] - opening_center[1], opening_target[0] - opening_center[0])
                    recent_explore_angles.append(angle)
                    recent_explore_angles = recent_explore_angles[-6:]
                    last_explore_target = opening_target

                cast_opening_buff_sequence(opening_dx, opening_dy)
                zone_opening_rightclick_pending   = False
                zone_opening_key3_taps_remaining  = ZONE_OPENING_KEY3_START_TAPS
                next_opening_key3_t               = time.time() + 0.5
                next_attack_rightclick_t          = time.time() + ATTACK_RIGHTCLICK_INTERVAL_SECONDS
                next_attack_key3_t                = time.time() + ATTACK_KEY3_INTERVAL_SECONDS
                sleep_interruptible(0.10)
                continue

            # ── capture minimap ───────────────────────────────────────────────
            minimap_region = get_minimap_region(minimap_mode)
            minimap        = cv2.cvtColor(np.array(sct.grab(minimap_region)), cv2.COLOR_BGRA2BGR)

            if hp is not None:
                if hp <= HP_ZERO_DEAD_THRESHOLD:
                    if hp_zero_since == 0.0:
                        hp_zero_since = now_loop
                else:
                    hp_zero_since = 0.0
            else:
                if last_known_hp is None or last_known_hp > HP_ZERO_DEAD_THRESHOLD:
                    hp_zero_since = 0.0

            hp_zero_sustained = (
                hp_zero_since > 0.0
                and (now_loop - hp_zero_since) >= 1.5
                and now_loop >= hp_bar_missing_until
            )

            if hp_zero_sustained:
                if attack_held:
                    left_mouse_up()
                    attack_held = False
                release_all_inputs()
                if (now_loop - last_death_respawn_t) >= HP_ZERO_RESPAWN_COOLDOWN_SECONDS:
                    print("[DEATH-HP] HP 0% bestaetigt -> Respawn + Vote Reset")
                    last_death_respawn_t = now_loop
                    attempt_respawn(sct, full_mon)
                    sleep_interruptible(RESPAWN_TO_RESET_DELAY_SECONDS)
                    auto_esc_ignore_until = time.time() + 1.0
                    if run_vote_reset_sequence(sct, full_mon):
                        _do_zone_reset()
                sleep_interruptible(0.08)
                continue

            if hp is None and now_loop < hp_bar_missing_until:
                sleep_interruptible(0.05)
                continue

            hp_for_potion = hp if hp is not None else last_known_hp
            if hp_for_potion is not None:
                potion_window = (
                    HEALTH_POTION_BURST_MIN_HP <= hp_for_potion <= HEALTH_POTION_BURST_MAX_HP
                )
                hp_is_zero_like = hp_for_potion <= HP_ZERO_DEAD_THRESHOLD
                if potion_window and not hp_is_zero_like and \
                        (now_loop - last_health_potion_t) >= HEALTH_POTION_COOLDOWN_SECONDS:
                    for _ in range(HEALTH_POTION_BURST_COUNT):
                        tap_key(HEALTH_POTION_KEY, hold_time=0.06)
                        time.sleep(HEALTH_POTION_BURST_GAP_SECONDS)
                    last_health_potion_t = now_loop
                if (hp or 0.0) < HP_FLEE_THRESHOLD and state != State.FLEE:
                    print(f"HP {(hp or 0.0):.0%} -> FLEE")
                    state       = State.FLEE
                    state_start = time.time()

            # ── enemy detection ───────────────────────────────────────────────
            enemy_dots = classify_enemy_dots(minimap)
            dots       = [(x, y) for x, y, _, _ in enemy_dots]
            center = minimap_center_from_image(minimap)

            terrain_memory.update(minimap)

            dot_filter_mask = build_walkable_minimap_mask(minimap, terrain_memory=terrain_memory)
            if dot_filter_mask is not None:
                h_mm, w_mm = dot_filter_mask.shape[:2]
                dots = [d for d in dots
                        if 0 <= d[0] < w_mm and 0 <= d[1] < h_mm
                        and dot_filter_mask[d[1], d[0]] > 0]

            companion.update(dots, center)
            filtered_dots = companion.filter(dots)
            if len(filtered_dots) == 1 and dot_distance(filtered_dots[0], center) <= (COMPANION_TRACK_RADIUS * 0.85):
                filtered_dots = []

            tier_by_dot = {}
            for x, y, tier, kind in enemy_dots:
                key = (x, y)
                tier_by_dot[key] = max(tier_by_dot.get(key, 0), tier)

            filtered_enemy_dots = [
                (dot[0], dot[1], tier_by_dot.get((dot[0], dot[1]), 1))
                for dot in filtered_dots
            ]

            cached_mask = buffered_walkable_mask(minimap, center)
            if cached_mask is not None:
                cached_mask = terrain_memory.apply(cached_mask)

            if len(filtered_enemy_dots) > 1 and cached_mask is not None:
                def _dot_score(entry):
                    dot = (entry[0], entry[1])
                    tier = entry[2]
                    return ray_walkability_score(cached_mask, center, dot, samples=12) * 100 \
                           - dot_distance(dot, center) * 0.3 + tier * 120.0
                nearest_entry = max(filtered_enemy_dots, key=_dot_score)
                nearest = (nearest_entry[0], nearest_entry[1])
            else:
                if filtered_enemy_dots:
                    nearest_entry = max(filtered_enemy_dots, key=lambda entry: (entry[2], -dot_distance((entry[0], entry[1]), center)))
                    nearest = (nearest_entry[0], nearest_entry[1])
                else:
                    nearest = None

            if nearest is not None:
                enemy_visible_frames = min(enemy_visible_frames + 1,
                                           ENEMY_CONFIRM_FRAMES + ENEMY_LOST_FRAMES + 6)
                enemy_lost_frames    = 0
            else:
                enemy_lost_frames += 1
                if enemy_lost_frames >= ENEMY_LOST_FRAMES:
                    enemy_visible_frames = 0

            confirmed_enemy_now = enemy_visible_frames >= ENEMY_CONFIRM_FRAMES

            if confirmed_enemy_now and nearest is not None:
                last_seen_enemy_dot = nearest
                vx = nearest[0] - center[0]
                vy = nearest[1] - center[1]
                if math.hypot(vx, vy) >= 2.0:
                    last_seen_enemy_vec = (vx, vy)
                last_seen_enemy_t = now_loop

            if state == State.ATTACK and confirmed_enemy_now:
                zone_had_attack        = True
                last_attack_activity_t = now_loop

            if confirmed_enemy_now:
                last_enemy_seen_t    = now_loop
                stuck_no_enemy_count = 0
                no_enemy_frames      = 0
            else:
                no_enemy_frames += 1

            reset_enemy_clear = (
                no_enemy_frames >= NO_ENEMY_RESET_MIN_FRAMES
                and (now_loop - last_enemy_seen_t) >= VOTE_RESET_ENEMY_GRACE_SECONDS
            )
            zone_reset_armed = (now_loop - zone_started_t) >= ZONE_MIN_EXPLORE_BEFORE_RESET_SECONDS

            reset_safe_window = (
                state == State.EXPLORE
                and not confirmed_enemy_now
                and reset_enemy_clear
                and (hp or 0.0) >= VOTE_RESET_MIN_HP
                and zone_reset_armed
            )
            stuck_reset_safe_window = (
                (
                    state == State.EXPLORE
                    or (
                        state == State.ATTACK
                        and no_enemy_frames >= (NO_ENEMY_RESET_MIN_FRAMES + 10)
                        and (now_loop - last_enemy_seen_t) >= (VOTE_RESET_ENEMY_GRACE_SECONDS + 2.0)
                    )
                )
                and not confirmed_enemy_now
                and reset_enemy_clear
                and (hp or 0.0) >= VOTE_RESET_MIN_HP
                and zone_reset_armed
            )

            hard_stuck_age     = (now_loop - hard_stuck_since) if hard_stuck_since > 0.0 else 0.0
            hard_stuck_trigger = hard_stuck_since > 0.0 and hard_stuck_age >= HARD_STUCK_RESET_SECONDS

            if hard_stuck_trigger and \
                    (now_loop - last_hard_stuck_snapshot_t) >= HARD_STUCK_SNAPSHOT_COOLDOWN_SECONDS:
                save_hard_stuck_snapshot(frame, minimap, nearest, center, hard_stuck_age)
                last_hard_stuck_snapshot_t = now_loop

            if debug_mode and (now_loop - last_reset_diag_t) >= RESET_DIAG_PRINT_INTERVAL_SECONDS:
                cooldown_ready = (now_loop - last_vote_reset_t) >= VOTE_RESET_COOLDOWN_SECONDS
                print(
                    "[RST-DIAG] "
                    f"state={state.value} nearest={nearest is not None} conf={confirmed_enemy_now} "
                    f"hp={hp:.2f}/{VOTE_RESET_MIN_HP:.2f} "
                    f"zone_age={now_loop - zone_started_t:.1f}s/{ZONE_MIN_EXPLORE_BEFORE_RESET_SECONDS:.1f}s "
                    f"no_enemy_frames={no_enemy_frames}/{NO_ENEMY_RESET_MIN_FRAMES} "
                    f"enemy_age={now_loop - last_enemy_seen_t:.2f}s/{VOTE_RESET_ENEMY_GRACE_SECONDS:.2f}s "
                    f"hard_stuck={hard_stuck_age:.1f}s/{HARD_STUCK_RESET_SECONDS:.1f}s "
                    f"cooldown={'ok' if cooldown_ready else 'wait'} "
                    f"stuck_chain={stuck_no_enemy_count}/{STUCK_VOTE_RESET_CONSECUTIVE}"
                )
                last_reset_diag_t = now_loop

            # ── vote-reset triggers ───────────────────────────────────────────
            cooldown_ok = (now_loop - last_vote_reset_t) >= VOTE_RESET_COOLDOWN_SECONDS

            if hard_stuck_trigger and confirmed_enemy_now and cooldown_ok:
                print(f"[STUCK-ENEMY] Stuck {hard_stuck_age:.1f}s with visible enemy -> spin + Vote Reset")
                release_all_inputs()
                attack_held = False
                spin_to_clear_melee()
                auto_esc_ignore_until = time.time() + 1.0
                if run_vote_reset_sequence(sct, full_mon):
                    _do_zone_reset()
                continue

            if zone_had_attack and reset_safe_window and cooldown_ok and \
                    (now_loop - last_attack_activity_t) >= ATTACK_IDLE_VOTE_RESET_SECONDS:
                if attack_held:
                    left_mouse_up()
                    attack_held = False
                print("Keine Attacke seit 5s -> spin + Vote Reset")
                spin_to_clear_melee()
                auto_esc_ignore_until = time.time() + 1.0
                if run_vote_reset_sequence(sct, full_mon):
                    _do_zone_reset()
                continue

            if reset_safe_window and cooldown_ok and \
                    (now_loop - zone_started_t) >= ZONE_MAX_SECONDS_BEFORE_RESET:
                if attack_held:
                    left_mouse_up()
                    attack_held = False
                print(f"Zone > {ZONE_MAX_SECONDS_BEFORE_RESET:.0f}s ohne verwertbaren Kampf -> spin + Vote Reset")
                spin_to_clear_melee()
                auto_esc_ignore_until = time.time() + 1.0
                if run_vote_reset_sequence(sct, full_mon):
                    _do_zone_reset()
                continue

            emergency_stuck = (
                hard_stuck_trigger
                and (hp or 0.0) < VOTE_RESET_MIN_HP
                and cooldown_ok
                and (now_loop - last_death_respawn_t) >= HP_ZERO_RESPAWN_COOLDOWN_SECONDS
            )
            if emergency_stuck:
                print(f"[STUCK-DEAD] Hard-stuck {hard_stuck_age:.1f}s + low HP -> Respawn + Vote Reset")
                release_all_inputs()
                attack_held          = False
                last_death_respawn_t = now_loop
                if attempt_respawn(sct, full_mon) and sleep_interruptible(RESPAWN_TO_RESET_DELAY_SECONDS):
                    auto_esc_ignore_until = time.time() + 1.0
                    if run_vote_reset_sequence(sct, full_mon):
                        _do_zone_reset()
                continue

            if stuck_reset_safe_window and cooldown_ok and \
                    (stuck_no_enemy_count >= STUCK_VOTE_RESET_CONSECUTIVE or hard_stuck_trigger):
                if hard_stuck_trigger and \
                        corner_escape_attempts < CORNER_ESCAPE_MAX_ATTEMPTS_BEFORE_RESET and \
                        (now_loop - last_corner_escape_t) >= CORNER_ESCAPE_COOLDOWN_SECONDS:
                    print(f"Hard-Stuck {hard_stuck_age:.1f}s -> Corner Escape Burst "
                          f"{corner_escape_attempts + 1}/{CORNER_ESCAPE_MAX_ATTEMPTS_BEFORE_RESET}")
                    if attack_held:
                        left_mouse_up()
                        attack_held = False
                    pref_dx = last_seen_enemy_vec[0] if last_seen_enemy_vec else None
                    pref_dy = last_seen_enemy_vec[1] if last_seen_enemy_vec else None
                    perform_corner_escape_burst(
                        explore_dx, explore_dy,
                        preferred_dx=pref_dx, preferred_dy=pref_dy,
                        minimap_bgr=minimap, center=center,
                    )
                    corner_escape_attempts += 1
                    last_corner_escape_t    = now_loop
                    hard_stuck_since        = now_loop
                    continue

                if attack_held:
                    left_mouse_up()
                    attack_held = False
                if hard_stuck_trigger:
                    print(f"Hard-Stuck {hard_stuck_age:.1f}s ohne Gegner -> Vote Reset")
                else:
                    print(f"{stuck_no_enemy_count}x stuck ohne Gegner -> Vote Reset")
                auto_esc_ignore_until = time.time() + 1.0
                if run_vote_reset_sequence(sct, full_mon):
                    _do_zone_reset()
                continue

            # ── debug frame save ──────────────────────────────────────────────
            if debug_mode:
                companion_dot = companion._candidate \
                    if companion._hits >= companion.confirm_frames else None
                save_minimap_frame(minimap, dots, companion_dot,
                                   filtered_dots, nearest, center, debug_frame_idx,
                                   state=state.value, walkable_mask=cached_mask,
                                   explore_target=last_explore_target)
                debug_frame_idx += 1
                if debug_frame_idx % 10 == 0:
                    print(f"[DBG] frame={debug_frame_idx} raw={len(dots)} "
                          f"filtered={len(filtered_dots)} nearest={nearest} state={state.value}")

            # ── right-click shield refresh ────────────────────────────────────
            if now_loop >= next_attack_rightclick_t and (now_loop - last_jump_t[0]) >= 0.42:
                attack_held              = cast_right_click_skill(attack_held, resume_left=False)
                rightclick_busy_until    = time.time() + 0.35
                next_attack_rightclick_t = now_loop + ATTACK_RIGHTCLICK_INTERVAL_SECONDS

            # ── enemy triggers ATTACK state ───────────────────────────────────
            if state != State.FLEE and confirmed_enemy_now and nearest:
                adx, ady       = dot_to_screen_delta(nearest, center)
                last_flee_dir  = (-adx, -ady)
                last_enemy_seen_t = time.time()
                if state != State.ATTACK:
                    print("Gegner gefunden -> ATTACK (Prio)")
                    state            = State.ATTACK
                    state_start      = time.time()
                    hard_stuck_since = 0.0
                    next_attack_rightclick_t = now_loop + ATTACK_RIGHTCLICK_INTERVAL_SECONDS
                    next_attack_key3_t       = now_loop + ATTACK_KEY3_INTERVAL_SECONDS
                    last_attack_sweep_t      = now_loop - ATTACK_SWEEP_INTERVAL_SECONDS
                    continue

            # ── opening key-3 drip taps ───────────────────────────────────────
            if zone_opening_key3_taps_remaining > 0 and now_loop >= next_opening_key3_t:
                tap_key('3', hold_time=0.12)
                zone_opening_key3_taps_remaining -= 1
                next_opening_key3_t = now_loop + 0.40

            # ── STATE: EXPLORE ────────────────────────────────────────────────
            if state == State.EXPLORE:
                if attack_held:
                    left_mouse_up()
                    attack_held = False

                if time.time() < explore_pause_until:
                    sleep_interruptible(0.05)
                    continue

                pt = ctypes.wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                stuck.update(pt.x, pt.y)
                explore_heatmap.update(center, minimap.shape[:2])

                replanning_due   = (time.time() - last_explore_plan_t) > EXPLORE_REPLAN_SECONDS
                explore_is_stuck = stuck.is_stuck()

                if explore_is_stuck:
                    now = time.time()
                    if not confirmed_enemy_now and hard_stuck_since == 0.0:
                        hard_stuck_since = now
                    if (now - last_stuck_log_t) > STUCK_LOG_COOLDOWN:
                        print("Stuck – neue Minimap-Richtung")
                        last_stuck_log_t = now
                    if not confirmed_enemy_now and \
                            (now - last_stuck_count_t) >= STUCK_COUNT_EVENT_COOLDOWN_SECONDS:
                        stuck_no_enemy_count += 1
                        last_stuck_count_t    = now
                    backtrack_move(explore_dx, explore_dy, last_backtrack_t)
                    if now_loop >= rightclick_busy_until:
                        maybe_jump_with_movement(explore_dx, explore_dy, last_jump_t, force_aim=True)
                    blocked_explore_angles.append(math.atan2(explore_dy, explore_dx))
                    blocked_explore_angles = blocked_explore_angles[-8:]
                    replanning_due = True
                else:
                    hard_stuck_since       = 0.0
                    corner_escape_attempts = 0

                if replanning_due:
                    recent_enemy_valid = (now_loop - last_seen_enemy_t) <= LAST_SEEN_MONSTER_MEMORY_SECONDS
                    target = None
                    if recent_enemy_valid:
                        (guided_move, target) = choose_recent_enemy_delta(
                            minimap, center,
                            recent_enemy_dot=last_seen_enemy_dot,
                            recent_enemy_vec=last_seen_enemy_vec,
                            terrain_memory=terrain_memory,
                        )
                        if guided_move is not None:
                            explore_dx, explore_dy = guided_move
                    if target is None:
                        (explore_dx, explore_dy), target = choose_explore_delta(
                            minimap, center, recent_explore_angles, blocked_explore_angles,
                            heatmap=explore_heatmap, terrain_memory=terrain_memory)
                    if target is not None:
                        angle = math.atan2(target[1] - center[1], target[0] - center[0])
                        recent_explore_angles.append(angle)
                        recent_explore_angles = recent_explore_angles[-6:]
                        last_explore_target = target
                    last_explore_plan_t = time.time()

                if cached_mask is not None:
                    walk_score = ray_walkability_score(
                        cached_mask, center,
                        (center[0] + explore_dx, center[1] + explore_dy), samples=14)
                    if walk_score < 0.55:
                        recent_enemy_valid = (now_loop - last_seen_enemy_t) <= LAST_SEEN_MONSTER_MEMORY_SECONDS
                        target = None
                        if recent_enemy_valid:
                            (guided_move, target) = choose_recent_enemy_delta(
                                minimap, center,
                                recent_enemy_dot=last_seen_enemy_dot,
                                recent_enemy_vec=last_seen_enemy_vec,
                                terrain_memory=terrain_memory,
                            )
                            if guided_move is not None:
                                explore_dx, explore_dy = guided_move
                        if target is None:
                            (explore_dx, explore_dy), target = choose_explore_delta(
                                minimap, center, recent_explore_angles, blocked_explore_angles,
                                heatmap=explore_heatmap, terrain_memory=terrain_memory)
                        if target is not None:
                            angle = math.atan2(target[1] - center[1], target[0] - center[0])
                            recent_explore_angles.append(angle)
                            recent_explore_angles = recent_explore_angles[-6:]
                            last_explore_target = target
                        last_explore_plan_t = time.time()

                wasd_move(explore_dx, explore_dy,
                          random.uniform(EXPLORE_MOVE_MIN_SECONDS, EXPLORE_MOVE_MAX_SECONDS))
                if now_loop >= rightclick_busy_until:
                    maybe_jump_with_movement(explore_dx, explore_dy, last_jump_t)

            # ── STATE: ATTACK ─────────────────────────────────────────────────
            elif state == State.ATTACK:
                if not attack_held and now_loop >= rightclick_busy_until:
                    left_mouse_down()
                    attack_held = True

                if nearest and (now_loop - last_attack_sweep_t) >= ATTACK_SWEEP_INTERVAL_SECONDS:
                    perform_attack_sweep(
                        duration=ATTACK_SWEEP_DURATION_SECONDS,
                        radius=ATTACK_SWEEP_RADIUS,
                        move_dx=attack_move_dx,
                        move_dy=attack_move_dy,
                    )
                    last_attack_sweep_t = time.time()

                if now_loop >= next_attack_key3_t:
                    attack_held        = cast_key3_skill_safe(attack_held, resume_left=True)
                    next_attack_key3_t = now_loop + ATTACK_KEY3_INTERVAL_SECONDS
                    if debug_mode:
                        print("[CAST] key 3")

                attack_stuck.update(center[0], center[1])
                if attack_stuck.is_stuck():
                    attack_age = now_loop - state_start
                    if attack_age < ATTACK_STUCK_ARM_SECONDS:
                        if (now_loop - last_stuck_log_t) > STUCK_LOG_COOLDOWN:
                            print(f"[STUCK-ENEMY] Armed at {attack_age:.1f}s/{ATTACK_STUCK_ARM_SECONDS:.1f}s; holding off reset")
                            last_stuck_log_t = now_loop
                    else:
                        if hard_stuck_since == 0.0:
                            hard_stuck_since = now_loop
                        if not confirmed_enemy_now and \
                                (now_loop - last_stuck_count_t) >= STUCK_COUNT_EVENT_COOLDOWN_SECONDS:
                            stuck_no_enemy_count += 1
                            last_stuck_count_t    = now_loop
                    backtrack_move(attack_move_dx, attack_move_dy, last_backtrack_t)
                    if now_loop >= rightclick_busy_until:
                        if confirmed_enemy_now:
                            _aim_cursor_toward_vector(adx, ady, distance=400)
                            keyboard.press('space')
                            time.sleep(0.04)
                            keyboard.release('space')
                            last_jump_t[0] = now_loop
                        else:
                            maybe_jump_with_movement(attack_move_dx, attack_move_dy,
                                                     last_jump_t, force_aim=True)
                    perp_dx = -attack_move_dy
                    perp_dy =  attack_move_dx
                    if cached_mask is not None:
                        ss_pos = ray_walkability_score(
                            cached_mask, center,
                            (center[0] + perp_dx, center[1] + perp_dy), samples=10)
                        ss_neg = ray_walkability_score(
                            cached_mask, center,
                            (center[0] - perp_dx, center[1] - perp_dy), samples=10)
                        slide_dx = perp_dx if ss_pos >= ss_neg else -perp_dx
                        slide_dy = perp_dy if ss_pos >= ss_neg else -perp_dy
                    else:
                        slide_dx, slide_dy = perp_dx, perp_dy
                    wasd_move_and_attack(slide_dx, slide_dy, duration=0.18,
                                         aim_dx=adx, aim_dy=ady)
                elif confirmed_enemy_now:
                    hard_stuck_since       = 0.0
                    corner_escape_attempts = 0

                enemy_dist = None
                if confirmed_enemy_now and nearest:
                    last_enemy_seen_t = time.time()
                    adx, ady          = dot_to_screen_delta(nearest, center)
                    enemy_dist        = dot_distance(nearest, center)
                    last_flee_dir     = (-adx, -ady)

                enemy_gone = (time.time() - last_enemy_seen_t) > ATTACK_LOST_GRACE
                timed_out  = (time.time() - state_start) > ATTACK_TIMEOUT

                if enemy_gone:
                    if attack_held:
                        left_mouse_up()
                        attack_held = False
                    if ENABLE_LOOT:
                        print("Gegner tot -> LOOT")
                        state = State.LOOT
                    else:
                        print("Gegner tot -> EXPLORE")
                        state = State.EXPLORE
                        explore_pause_until = time.time() + random.uniform(
                            POST_KILL_SCAN_MIN_SECONDS, POST_KILL_SCAN_MAX_SECONDS)
                    state_start = time.time()
                    continue

                if timed_out and confirmed_enemy_now:
                    state_start = time.time()
                elif timed_out and not confirmed_enemy_now:
                    if attack_held:
                        left_mouse_up()
                        attack_held = False
                    print("ATTACK Timeout ohne Sichtkontakt -> EXPLORE")
                    state       = State.EXPLORE
                    state_start = time.time()
                    continue

                if not confirmed_enemy_now and \
                        (now_loop - last_enemy_seen_t) >= ATTACK_NO_CONFIRMED_ENEMY_TIMEOUT_SECONDS:
                    if attack_held:
                        left_mouse_up()
                        attack_held = False
                    print("ATTACK ohne bestaetigten Gegner -> EXPLORE")
                    state       = State.EXPLORE
                    state_start = time.time()
                    continue

                if (time.time() - last_attack_move_plan_t) > ATTACK_MOVE_REPLAN_SECONDS:
                    (attack_move_dx, attack_move_dy), attack_strafe_sign, \
                    kite_committed_until, kite_cached_move = choose_attack_move_delta(
                        adx, ady, minimap, center, recent_explore_angles,
                        attack_strafe_sign, enemy_dist,
                        kite_committed_until=kite_committed_until,
                        kite_cached_move=kite_cached_move,
                        terrain_memory=terrain_memory,
                    )
                    last_attack_move_plan_t = time.time()

                wasd_move_and_attack(
                    attack_move_dx, attack_move_dy,
                    duration=random.uniform(0.95, 1.45),
                    aim_dx=adx, aim_dy=ady,
                )
                if now_loop >= rightclick_busy_until:
                    maybe_jump_with_movement(attack_move_dx, attack_move_dy, last_jump_t)

            # ── STATE: LOOT ───────────────────────────────────────────────────
            elif state == State.LOOT:
                if attack_held:
                    left_mouse_up()
                    attack_held = False

                if (time.time() - state_start) > LOOT_TIMEOUT:
                    state       = State.EXPLORE
                    state_start = time.time()
                    continue

                item = find_items_on_screen(frame, item_templates)
                if item:
                    print(f"Item {item} -> pickup")
                    ctypes.windll.user32.SetCursorPos(item[0], item[1])
                    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                    sleep_interruptible(0.05)
                    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                    sleep_interruptible(2.0)
                else:
                    state       = State.EXPLORE
                    state_start = time.time()

            # ── STATE: FLEE ───────────────────────────────────────────────────
            elif state == State.FLEE:
                if nearest:
                    adx, ady = dot_to_screen_delta(nearest, center)
                    _aim_cursor_toward_vector(adx, ady)
                    if not attack_held:
                        left_mouse_down()
                        attack_held = True
                elif attack_held:
                    left_mouse_up()
                    attack_held = False

                flee_delta = None
                if cached_mask is not None:
                    lx, ly   = last_flee_dir if last_flee_dir != (0, 0) else random_explore_delta()
                    flee_len = math.hypot(lx, ly)
                    if flee_len > 1e-6:
                        cx_mm, cy_mm = center
                        flee_target  = (
                            int(max(0, min(minimap.shape[1] - 1, cx_mm + (lx / flee_len) * 40))),
                            int(max(0, min(minimap.shape[0] - 1, cy_mm + (ly / flee_len) * 40))),
                        )
                        waypoint = _a_star_next_waypoint(cached_mask, center, flee_target)
                        if waypoint is not None:
                            flee_delta = dot_to_screen_delta(waypoint, center)
                if flee_delta is None:
                    fx, fy     = last_flee_dir if last_flee_dir != (0, 0) else random_explore_delta()
                    flee_delta = (fx, fy)
                wasd_move(flee_delta[0], flee_delta[1], random.uniform(0.5, 1.0))

                hp = get_hp_percent(frame)
                if hp is None:
                    if (time.time() - state_start) > FLEE_MAX_WAIT:
                        print("HP missing -> EXPLORE")
                        state       = State.EXPLORE
                        state_start = time.time()
                elif hp >= HP_RESUME_THRESHOLD or (time.time() - state_start) > FLEE_MAX_WAIT:
                    print(f"HP {hp:.0%} -> EXPLORE")
                    state       = State.EXPLORE
                    state_start = time.time()

            sleep_interruptible(random.uniform(0.05, 0.12))

    print("Bot beendet.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("Bot per Ctrl+C gestoppt.")
