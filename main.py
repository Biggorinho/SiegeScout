import ctypes
import ctypes.wintypes
import cv2
import mss
import numpy as np
import random
import time
import math
import keyboard
import stats_tracker as _st

from config import (
    SCREEN_W, SCREEN_H,
    HP_FLEE_THRESHOLD, HP_RESUME_THRESHOLD,
    HP_ZERO_DEAD_THRESHOLD, HP_ZERO_RESPAWN_COOLDOWN_SECONDS,
    HP_BAR_MISSING_GRACE_SECONDS,
    HEALTH_POTION_KEY, HEALTH_POTION_COOLDOWN_SECONDS,
    HEALTH_POTION_BURST_MIN_HP, HEALTH_POTION_BURST_MAX_HP,
    HEALTH_POTION_BURST_COUNT, HEALTH_POTION_BURST_GAP_SECONDS,
    HP_RECOVERY_THRESHOLD,
    HP_RECOVERY_POTION_PRESSES, HP_RECOVERY_COOLDOWN_SECONDS,
    ATTACK_LOST_GRACE, ATTACK_TIMEOUT, FLEE_MAX_WAIT,
    ATTACK_MOVE_REPLAN_SECONDS, ATTACK_CHASE_MINIMAP_DIST,
    ATTACK_RIGHTCLICK_INTERVAL_SECONDS, ATTACK_KEY3_INTERVAL_SECONDS,
    ATTACK_SWEEP_INTERVAL_SECONDS, ATTACK_SWEEP_DURATION_SECONDS, ATTACK_SWEEP_RADIUS,
    ATTACK_NO_CONFIRMED_ENEMY_TIMEOUT_SECONDS,
    ATTACK_STUCK_WINDOW, ATTACK_STUCK_MIN_MOVE,
    ATTACK_STUCK_ARM_SECONDS,
    EXPLORE_MOVE_MIN_SECONDS, EXPLORE_MOVE_MAX_SECONDS,
    EXPLORE_STUCK_WINDOW, EXPLORE_STUCK_MIN_MOVE,
    TERRAIN_BLOCKED_MARK_COOLDOWN_SECONDS,
    POST_KILL_SCAN_MIN_SECONDS, POST_KILL_SCAN_MAX_SECONDS,
    STUCK_LOG_COOLDOWN, STUCK_COUNT_EVENT_COOLDOWN_SECONDS,
    STUCK_VOTE_RESET_CONSECUTIVE, HARD_STUCK_RESET_SECONDS,
    CORNER_ESCAPE_COOLDOWN_SECONDS, CORNER_ESCAPE_MAX_ATTEMPTS_BEFORE_RESET,
    VOTE_RESET_COOLDOWN_SECONDS, VOTE_RESET_FAIL_COOLDOWN_SECONDS, VOTE_RESET_ENEMY_GRACE_SECONDS,
    VOTE_RESET_FAIL_STREAK_LOCKOUT_COUNT, VOTE_RESET_FAIL_LOCKOUT_SECONDS,
    VOTE_RESET_MIN_HP, NO_ENEMY_RESET_MIN_FRAMES,
    ZONE_MAX_SECONDS_BEFORE_RESET, ZONE_MIN_EXPLORE_BEFORE_RESET_SECONDS,
    ZONE_TRANSITION_FREEZE_SECONDS,
    ZONE_PACKET_MIN_FREEZE_SECONDS,
    ZONE_PACKET_SETTLE_SECONDS,
    DEATH_RESPAWN_COOLDOWN_SECONDS, DEATH_CONFIRM_FRAMES,
    RESPAWN_TO_RESET_DELAY_SECONDS,
    ENEMY_CONFIRM_FRAMES, ENEMY_LOST_FRAMES,
    LAST_SEEN_MONSTER_MEMORY_SECONDS,
    COMPANION_TRACK_RADIUS,
    ATTACK_IDLE_VOTE_RESET_SECONDS,
    ESCAPE_COMMIT_MIN_SECONDS, ESCAPE_COMMIT_MAX_SECONDS,
    MINING_SCAN_INTERVAL_SECONDS,
    MINING_CLICK_INTERVAL_SECONDS,
    MINING_INTERACT_INTERVAL_SECONDS,
    MINING_QTE_PRESS_COOLDOWN_SECONDS,
    LOOT_ITEMS_SCAN_INTERVAL_SECONDS,
    LOOT_ITEMS_CLICK_INTERVAL_SECONDS,
    MINING_APPROACH_MIN_DIST_PX,
    MINING_APPROACH_JUMP_DIST_PX,
    MINING_APPROACH_MOVE_MIN_SECONDS,
    MINING_APPROACH_MOVE_MAX_SECONDS,
    MINING_SESSION_MEMORY_SECONDS,
    RESOURCE_COLLECT_START_DELAY_SECONDS,
    ENABLE_MINIMAP_RESOURCE_TARGETING,
    MINIMAP_RESOURCE_PICKUP_DIST_PX,
    MINIMAP_RESOURCE_VISIT_MEMORY_SECONDS,
    MINIMAP_RESOURCE_MAX_TRAVEL_SECONDS,
    MINIMAP_RESOURCE_SPOT_MATCH_RADIUS_PX,
    MINIMAP_RESOURCE_CLOSE_ATTEMPT_SECONDS,
    ENABLE_DEBUG_SNAPSHOTS,
    DEBUG_DIR,
    DEBUG_PERIODIC_INTERVAL_SECONDS,
    DEBUG_MAX_FILES,
    DEBUG_EVENT_RATE_LIMIT_SECONDS,
    DEBUG_MOSAIC_PANEL_HEIGHT_PX,
)
from detectors import (
    classify_enemy_dots, get_hp_percent,
    is_death_screen, is_world_map_visible,
    is_pause_menu_visible, is_options_menu_visible,
    find_ore_nodes_on_screen, detect_mining_qte_window,
    find_item_tags_on_screen,
    classify_item_dots, classify_ore_dots,
)
from minimap import (
    get_minimap_region, minimap_center_from_image,
    dot_distance, dot_to_screen_delta,
    build_walkable_minimap_mask, buffered_walkable_mask,
    random_explore_delta,
)
from pathfinding import (
    ray_walkability_score, choose_explore_delta, choose_recent_enemy_delta,
    choose_attack_move_delta, _a_star_next_waypoint, octant_escape_delta,
    minimap_explore_target,
)
from state import State, StuckDetector, IntentStuckDetector, ExploreHeatMap, TerrainMemory, CompanionSuppressor, SectorTracker, EnemyChaser, ExploreNavigator, ResourceTargeter
from input_control import (
    sleep_interruptible, wasd_move, backtrack_move,
    explore_move_start, explore_move_stop,
    attack_key_down, attack_key_up,
    cast_key_skill_safe, cast_key3_skill_safe, cast_movement_ability,
    spin_to_clear_melee,
    tap_key, release_all_inputs,
    left_click_at,
    _aim_cursor_toward_vector, maybe_jump_with_movement,
    maybe_jump_stuck_sequence,
    wasd_move_and_attack, perform_attack_sweep,
)
from combat import (
    run_vote_reset_sequence, attempt_respawn, perform_corner_escape_burst,
    pick_zone_from_open_map,
)
from debug_utils import DebugRecorder


def run(
    stop_event=None, enable_death_handling=None, enable_mining=None, enable_loot_items=None,
    on_vote_reset=None, target_zone=None,
    enable_rightclick_skill=None, rightclick_cooldown=None,
    enable_key3_skill=None, key3_cooldown=None,
    enable_movement_ability=None, movement_ability_cooldown=None,
    enable_debug_snapshots=None,
):
    import config as _cfg
    _enable_death_handling   = _cfg.ENABLE_DEATH_HANDLING if enable_death_handling is None else enable_death_handling
    _enable_mining           = _cfg.ENABLE_MINING         if enable_mining is None         else enable_mining
    _enable_loot_items       = _cfg.ENABLE_LOOT_ITEMS     if enable_loot_items is None     else enable_loot_items
    _enable_debug_snapshots  = _cfg.ENABLE_DEBUG_SNAPSHOTS if enable_debug_snapshots is None else enable_debug_snapshots
    _enable_buff_shield  = _cfg.ENABLE_BUFF_SHIELD_SKILL if enable_rightclick_skill is None else enable_rightclick_skill
    _buff_shield_key     = _cfg.BUFF_SHIELD_KEY
    _buff_shield_cd      = ATTACK_RIGHTCLICK_INTERVAL_SECONDS if rightclick_cooldown is None else rightclick_cooldown
    _enable_key3_skill       = _cfg.ENABLE_KEY3_SKILL if enable_key3_skill is None else enable_key3_skill
    _key3_cooldown           = ATTACK_KEY3_INTERVAL_SECONDS if key3_cooldown is None else key3_cooldown
    _enable_movement_ability = _cfg.ENABLE_MOVEMENT_ABILITY if enable_movement_ability is None else enable_movement_ability
    _movement_ability_key    = _cfg.MOVEMENT_ABILITY_KEY
    _movement_ability_cd     = _cfg.MOVEMENT_ABILITY_COOLDOWN_SECONDS if movement_ability_cooldown is None else movement_ability_cooldown

    print("Bot started | q=Stop | p=Pause | esc=Emergency stop")
    print("Game window must be visible (no background capture with mss)")
    print("Minimap mode: small")

    ENEMY_SOFT_RESET_GRACE_SECONDS = 3.0

    state              = State.EXPLORE
    paused             = False
    stuck              = StuckDetector(window=EXPLORE_STUCK_WINDOW, min_move=EXPLORE_STUCK_MIN_MOVE)
    explore_intent_stuck = IntentStuckDetector(window=10, intent_min=80.0, min_motion_px=0.60, stuck_ratio=0.70)
    attack_intent_stuck  = IntentStuckDetector(window=10, intent_min=90.0, min_motion_px=0.55, stuck_ratio=0.70)
    state_start        = time.time()
    last_flee_dir      = (0, 0)
    last_enemy_seen_t  = 0.0
    adx, ady           = 0, 0
    explore_dx, explore_dy       = random_explore_delta()
    explore_held_keys            = frozenset()
    recent_explore_angles        = []
    blocked_explore_angles       = []
    attack_held                  = False
    last_stuck_log_t             = 0.0
    attack_strafe_sign           = 1
    last_attack_move_plan_t      = 0.0
    attack_move_dx, attack_move_dy = random_explore_delta()
    explore_pause_until          = 0.0
    last_jump_t                  = [0.0]
    explore_stuck_jump_seq_idx   = [0]
    attack_stuck_jump_seq_idx    = [0]
    explore_stuck_jump_pref      = [False]
    attack_stuck_jump_pref       = [False]
    explore_stuck_active         = False
    attack_stuck_active          = False
    kite_committed_until         = 0.0
    kite_cached_move             = None
    last_backtrack_t             = [0.0]
    explore_heatmap              = ExploreHeatMap()
    explore_navigator            = ExploreNavigator()
    resource_targeter            = ResourceTargeter(
        memory_seconds=MINIMAP_RESOURCE_VISIT_MEMORY_SECONDS,
        max_travel_seconds=MINIMAP_RESOURCE_MAX_TRAVEL_SECONDS,
        spot_match_radius_px=MINIMAP_RESOURCE_SPOT_MATCH_RADIUS_PX,
        close_attempt_seconds=MINIMAP_RESOURCE_CLOSE_ATTEMPT_SECONDS,
    )
    debug_recorder               = DebugRecorder(
        enabled=_enable_debug_snapshots,
        debug_dir=DEBUG_DIR,
        periodic_interval=DEBUG_PERIODIC_INTERVAL_SECONDS,
        max_files=DEBUG_MAX_FILES,
        event_rate_limit_seconds=DEBUG_EVENT_RATE_LIMIT_SECONDS,
        panel_height=DEBUG_MOSAIC_PANEL_HEIGHT_PX,
    )
    last_periodic_debug_t        = 0.0
    enemy_chaser                 = EnemyChaser()
    sector_tracker               = SectorTracker()
    escape_committed_until       = 0.0
    escape_dx, escape_dy         = 0, 0
    terrain_memory               = TerrainMemory()
    attack_stuck                 = StuckDetector(window=ATTACK_STUCK_WINDOW, min_move=ATTACK_STUCK_MIN_MOVE)
    cached_mask                  = None
    companion                    = CompanionSuppressor()
    minimap_mode                 = "small"
    last_attack_activity_t       = time.time()
    zone_had_attack              = False
    last_vote_reset_t            = 0.0
    vote_reset_fail_streak       = 0
    vote_reset_lockout_until     = 0.0
    auto_esc_ignore_until        = 0.0
    vote_reset_in_progress       = False
    menu_escape_cooldown_until   = 0.0
    auto_q_ignore_until          = 0.0
    zone_transition_until        = 0.0
    last_health_potion_t         = 0.0
    last_hp_recovery_t           = 0.0
    last_attack_key_refresh_t    = 0.0
    last_known_hp                = None
    next_attack_rightclick_t     = time.time() + _buff_shield_cd
    next_attack_key3_t           = time.time() + _key3_cooldown
    last_attack_sweep_t          = time.time()
    last_movement_ability_t      = 0.0
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
    last_seen_enemy_dot          = None
    last_seen_enemy_vec          = None
    last_seen_enemy_t            = 0.0
    last_any_enemy_seen_t        = 0.0
    _log_enemy_count             = -1
    _log_chase_t                 = 0.0
    zone_started_t               = time.time()
    last_death_respawn_t         = 0.0
    death_screen_frames          = 0
    zone_opening_rightclick_pending  = True
    zone_opening_key3_taps_remaining = 0
    next_opening_key3_t          = 0.0
    rightclick_busy_until        = 0.0
    last_explore_target          = None
    last_mining_scan_t           = 0.0
    last_mining_click_t          = 0.0
    last_mining_interact_t       = 0.0
    last_mining_qte_press_t      = 0.0
    mining_active_until          = 0.0
    mining_target_screen         = None
    last_loot_items_scan_t       = 0.0
    last_loot_items_click_t      = 0.0
    last_terrain_block_mark_t    = 0.0
    _zone_logged_reset_t         = 0.0   # last_vote_reset_t for which zone confirm was printed
    # ── shared reset helper ────────────────────────────────────────────────────
    # Defined as a closure so it can assign back into run()'s local scope.
    def _do_zone_reset():
        nonlocal last_vote_reset_t, zone_transition_until, last_attack_activity_t
        nonlocal zone_had_attack, stuck_no_enemy_count, hard_stuck_since
        nonlocal corner_escape_attempts, no_enemy_frames, hp_bar_missing_until
        nonlocal zone_started_t, zone_opening_rightclick_pending, state, state_start
        nonlocal explore_pause_until, death_screen_frames, hp_zero_since
        nonlocal explore_dx, explore_dy, last_explore_target
        nonlocal last_any_enemy_seen_t
        nonlocal escape_committed_until, escape_dx, escape_dy
        nonlocal explore_held_keys
        nonlocal explore_stuck_active, attack_stuck_active
        nonlocal explore_stuck_jump_seq_idx, attack_stuck_jump_seq_idx
        nonlocal explore_stuck_jump_pref, attack_stuck_jump_pref
        nonlocal mining_active_until, mining_target_screen
        nonlocal last_loot_items_scan_t, last_loot_items_click_t
        nonlocal last_terrain_block_mark_t
        nonlocal vote_reset_fail_streak, vote_reset_lockout_until
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
        last_any_enemy_seen_t         = 0.0
        explore_dx, explore_dy        = random_explore_delta()
        last_explore_target           = None

        # New zone means new topology; clear cross-zone memory to avoid stale bias.
        recent_explore_angles.clear()
        blocked_explore_angles.clear()
        explore_heatmap.reset()
        terrain_memory.reset()
        sector_tracker.reset()
        escape_committed_until = 0.0
        escape_dx, escape_dy   = 0, 0
        explore_held_keys      = explore_move_stop(explore_held_keys)
        explore_stuck_active   = False
        attack_stuck_active    = False
        explore_stuck_jump_seq_idx[0] = 0
        attack_stuck_jump_seq_idx[0] = 0
        explore_stuck_jump_pref[0] = False
        attack_stuck_jump_pref[0] = False
        explore_intent_stuck.reset()
        attack_intent_stuck.reset()
        mining_active_until = 0.0
        mining_target_screen = None
        last_loot_items_scan_t = 0.0
        last_loot_items_click_t = 0.0
        last_terrain_block_mark_t = 0.0
        vote_reset_fail_streak = 0
        vote_reset_lockout_until = 0.0
        explore_navigator.reset()
        resource_targeter.reset()

    def _enter_attack_mode(now_loop, adx=0, ady=0, hold_attack=False):
        nonlocal state, state_start, hard_stuck_since, escape_committed_until
        nonlocal next_attack_rightclick_t, next_attack_key3_t, last_attack_sweep_t
        nonlocal attack_move_dx, attack_move_dy, attack_strafe_sign
        nonlocal kite_committed_until, kite_cached_move, last_attack_move_plan_t
        nonlocal zone_opening_rightclick_pending, attack_held
        nonlocal attack_stuck_active, attack_stuck_jump_pref, attack_stuck_jump_seq_idx

        state                          = State.ATTACK
        state_start                    = now_loop
        hard_stuck_since               = 0.0
        escape_committed_until         = 0.0
        next_attack_rightclick_t       = now_loop + _buff_shield_cd
        next_attack_key3_t             = now_loop + _key3_cooldown
        last_attack_sweep_t            = now_loop - ATTACK_SWEEP_INTERVAL_SECONDS
        last_attack_move_plan_t        = 0.0
        attack_move_dx, attack_move_dy  = adx, ady
        attack_strafe_sign             = 1
        kite_committed_until           = 0.0
        kite_cached_move               = None
        zone_opening_rightclick_pending = False
        attack_stuck_active            = False
        attack_stuck_jump_pref[0]      = False
        attack_stuck_jump_seq_idx[0]   = 0
        if attack_held:
            attack_key_up()
            attack_held = False
        if hold_attack:
            attack_key_down()
            attack_held = True
        enemy_chaser.reset()

    with mss.mss() as sct:
        full_mon = {"top": 0, "left": 0, "width": SCREEN_W, "height": SCREEN_H}

        def _pre_reset_spin():
            nonlocal attack_held
            # Fixed pre-reset clear window to reduce resets while enemies are still nearby.
            release_all_inputs()
            attack_held = False
            print("[RST] pre-reset spin (~3s)")
            spin_to_clear_melee(rounds=2, radius=200, steps=36)

        def _run_vote_reset(target_zone):
            """Wrap run_vote_reset_sequence so the global safety-net flag is
            always cleared, even if the call raises."""
            nonlocal vote_reset_in_progress
            vote_reset_in_progress = True
            try:
                return run_vote_reset_sequence(sct, full_mon, target_zone=target_zone)
            finally:
                vote_reset_in_progress = False

        def _attempt_vote_reset(with_spin=True):
            """Run vote reset once and apply cooldown on both success and failure."""
            nonlocal last_vote_reset_t, auto_esc_ignore_until
            nonlocal vote_reset_fail_streak, vote_reset_lockout_until

            if with_spin:
                _pre_reset_spin()

            attempt_t = time.time()
            # Set immediately to prevent same-tick retriggers while reset is running.
            last_vote_reset_t = attempt_t
            auto_esc_ignore_until = attempt_t + 1.0
            ok = _run_vote_reset(target_zone)
            if ok:
                vote_reset_fail_streak = 0
                vote_reset_lockout_until = 0.0
                if on_vote_reset:
                    try:
                        on_vote_reset()
                    except Exception:
                        pass
                _do_zone_reset()
            else:
                vote_reset_fail_streak += 1
                fail_cd = max(0.0, min(VOTE_RESET_FAIL_COOLDOWN_SECONDS, VOTE_RESET_COOLDOWN_SECONDS))
                if vote_reset_fail_streak >= VOTE_RESET_FAIL_STREAK_LOCKOUT_COUNT:
                    vote_reset_lockout_until = max(vote_reset_lockout_until, attempt_t + VOTE_RESET_FAIL_LOCKOUT_SECONDS)
                    # Keep base cooldown math stable and enforce long backoff via lockout gate.
                    last_vote_reset_t = attempt_t - (VOTE_RESET_COOLDOWN_SECONDS - fail_cd)
                    print(
                        f"[RST] vote reset failed x{vote_reset_fail_streak} -> lockout "
                        f"{VOTE_RESET_FAIL_LOCKOUT_SECONDS:.0f}s (pause menu/click not reliable)"
                    )
                else:
                    # Keep short retry backoff without allowing rapid menu toggle loops.
                    last_vote_reset_t = attempt_t - (VOTE_RESET_COOLDOWN_SECONDS - fail_cd)
                    print(
                        f"[RST] vote reset failed x{vote_reset_fail_streak} -> cooldown {fail_cd:.0f}s "
                        "to avoid menu-toggle loop"
                    )
            return ok

        # ── startup: pick satanic zone from the world map ─────────────────────
        startup_frame = cv2.cvtColor(np.array(sct.grab(full_mon)), cv2.COLOR_BGRA2BGR)
        if is_world_map_visible(startup_frame):
            _zone_label = f"zone: {target_zone}" if target_zone else "satanic zone"
            print(f"[STARTUP] Map already open – selecting {_zone_label} ...")
            if pick_zone_from_open_map(sct, full_mon, startup_frame, target_zone=target_zone):
                _do_zone_reset()
        else:
            _zone_label = f"zone: {target_zone}" if target_zone else "satanic zone"
            print(f"[STARTUP] Opening map and selecting {_zone_label} ...")
            _attempt_vote_reset(with_spin=False)

        while True:
            now_loop = time.time()

            # Periodic debug snapshot — recorder gates by interval.
            if _enable_debug_snapshots and (now_loop - last_periodic_debug_t) >= DEBUG_PERIODIC_INTERVAL_SECONDS:
                _periodic_frame    = locals().get("frame")
                _periodic_minimap  = locals().get("minimap")
                _periodic_center   = locals().get("center", (0, 0))
                if _periodic_frame is not None and _periodic_minimap is not None:
                    debug_recorder.force_record(
                        now=now_loop, event="periodic",
                        frame_bgr=_periodic_frame, minimap_bgr=_periodic_minimap,
                        center=_periodic_center,
                        enemy_dots=[], item_dots=[], ore_dots=[],
                        companion_dot=None, nearest_enemy=None,
                        resource_target=resource_targeter.active,
                        walkable_mask=None, enemy_color_mask=None,
                        state_label=state.name if hasattr(state, "name") else str(state),
                        hp_percent=last_known_hp, counters_snapshot={},
                    )
                    last_periodic_debug_t = now_loop

            # ── hotkeys ───────────────────────────────────────────────────────
            if stop_event is not None and stop_event.is_set():
                release_all_inputs()
                attack_held = False
                print("Bot stopped (UI).")
                break
            if keyboard.is_pressed('esc') and now_loop > auto_esc_ignore_until:
                release_all_inputs()
                attack_held = False
                print("EMERGENCY STOP (esc)")
                break
            if (keyboard.is_pressed('f1') or keyboard.is_pressed('f8')) and now_loop > auto_q_ignore_until:
                if attack_held:
                    attack_key_up()
                    attack_held = False
                print("Bot stopped.")
                break
            if keyboard.is_pressed('p'):
                paused = not paused
                if paused and attack_held:
                    attack_key_up()
                    attack_held = False
                print("PAUSE" if paused else "RESUME")
                time.sleep(0.5)
                continue
            if paused:
                time.sleep(0.1)
                continue

            # Release held explore keys whenever we leave EXPLORE state
            if state != State.EXPLORE and explore_held_keys:
                explore_held_keys = explore_move_stop(explore_held_keys)

            # ── zone transition freeze ────────────────────────────────────────
            if now_loop < zone_transition_until:
                release_all_inputs()
                attack_held = False
                _pkt_tracker = _st.get_tracker()
                if (_pkt_tracker and _pkt_tracker.available
                        and last_vote_reset_t > _zone_logged_reset_t):
                    _za = _pkt_tracker.zone_entered_at
                    if _za is not None and _za > last_vote_reset_t:
                        _buffs = _pkt_tracker.active_buffs
                        _buff_str = ", ".join(_buffs) if _buffs else "none"
                        print(f"[ZONE] {_pkt_tracker.current_zone}  |  buffs: {_buff_str}")
                        _zone_logged_reset_t = last_vote_reset_t
                        # Early exit: if min freeze has passed, let the packet
                        # shorten the remaining wait instead of the fixed 5.5 s.
                        if (now_loop - last_vote_reset_t) >= ZONE_PACKET_MIN_FREEZE_SECONDS:
                            _early = _za + ZONE_PACKET_SETTLE_SECONDS
                            if _early < zone_transition_until:
                                _saved = zone_transition_until - max(_early, now_loop)
                                zone_transition_until = _early
                                print(f"[ZONE] Early freeze exit — saved {_saved:.1f}s")
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
            if _enable_death_handling and now_loop >= hp_bar_missing_until and _hp_critical:
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
                    if _run_vote_reset(target_zone):
                        _do_zone_reset()
                    death_screen_frames = 0
                    continue

            # ── zone opening buff sequence ────────────────────────────────────
            if zone_opening_rightclick_pending and now_loop >= zone_transition_until:
                release_all_inputs()
                attack_held = False
                auto_q_ignore_until = time.time() + 0.8
                # One-time zone packet check after freeze.
                # The packet often arrives slightly after the freeze ends on slower machines,
                # so wait up to 2 s before declaring a failure.
                _pkt_zo = _st.get_tracker()
                if (_pkt_zo and _pkt_zo.available
                        and last_vote_reset_t > _zone_logged_reset_t):
                    _za = _pkt_zo.zone_entered_at
                    if _za is None or _za <= last_vote_reset_t:
                        _wait_until = time.time() + 2.0
                        while time.time() < _wait_until:
                            if not sleep_interruptible(0.1):
                                break
                            _za = _pkt_zo.zone_entered_at
                            if _za is not None and _za > last_vote_reset_t:
                                break
                    if _za is not None and _za > last_vote_reset_t:
                        _buffs = _pkt_zo.active_buffs
                        _buff_str = ", ".join(_buffs) if _buffs else "none"
                        print(f"[ZONE] {_pkt_zo.current_zone}  |  buffs: {_buff_str}")
                    else:
                        print("[ZONE] no zone packet yet — zone selected, packet arrives on first game event")
                    _zone_logged_reset_t = last_vote_reset_t

                opening_frame = cv2.cvtColor(np.array(sct.grab(full_mon)), cv2.COLOR_BGRA2BGR)
                opening_hp = get_hp_percent(opening_frame)
                if opening_hp is None and (now_loop - zone_transition_until) < 4.0:
                    sleep_interruptible(0.08)
                    continue

                opening_region   = get_minimap_region(minimap_mode)
                opening_minimap  = cv2.cvtColor(np.array(sct.grab(opening_region)), cv2.COLOR_BGRA2BGR)
                opening_center   = minimap_center_from_image(opening_minimap)
                opening_dots = classify_enemy_dots(opening_minimap)

                if opening_dots:
                    opening_nearest = max(
                        opening_dots,
                        key=lambda entry: (entry[2], -dot_distance((entry[0], entry[1]), opening_center)),
                    )
                    adx, ady = dot_to_screen_delta((opening_nearest[0], opening_nearest[1]), opening_center)
                    print("[OPEN] enemy visible at zone start -> skip buff and attack")
                    _enter_attack_mode(now_loop, adx, ady, hold_attack=True)
                    continue

                # Seed initial explore direction from minimap pathfinding.
                opening_dx, opening_dy = explore_dx, explore_dy
                terrain_memory.update(opening_minimap)
                (opening_move, opening_target) = choose_explore_delta(
                    opening_minimap,
                    opening_center,
                    recent_explore_angles,
                    blocked_explore_angles,
                    heatmap=explore_heatmap,
                    terrain_memory=terrain_memory,
                    sector_tracker=sector_tracker,
                )
                if opening_move is not None:
                    opening_dx, opening_dy = opening_move
                    explore_dx, explore_dy = opening_move
                if opening_target is not None:
                    angle = math.atan2(opening_target[1] - opening_center[1], opening_target[0] - opening_center[0])
                    recent_explore_angles.append(angle)
                    recent_explore_angles = recent_explore_angles[-8:]
                    sector_tracker.update(angle)
                    last_explore_target = opening_target

                # Skip scripted opening spin/buff: start with normal explore pathing.
                zone_opening_rightclick_pending   = False
                zone_opening_key3_taps_remaining  = 0
                next_attack_rightclick_t          = time.time() + _buff_shield_cd
                next_attack_key3_t                = time.time() + _key3_cooldown
                continue

            # ── stuck-in-menu safety net ──────────────────────────────────────
            # Only fires when no vote-reset is in progress (vote-reset legitimately
            # opens the pause menu). Press ESC to exit Options menu first; on the
            # next iteration the loop will see the pause menu and ESC again.
            if not vote_reset_in_progress and now_loop >= menu_escape_cooldown_until:
                if is_options_menu_visible(frame):
                    print("[MENU] options menu detected — ESC")
                    auto_esc_ignore_until      = time.time() + 1.0
                    tap_key('esc', hold_time=0.12)
                    menu_escape_cooldown_until = now_loop + 0.8
                    continue
                if is_pause_menu_visible(frame):
                    print("[MENU] pause menu detected — ESC")
                    auto_esc_ignore_until      = time.time() + 1.0
                    tap_key('esc', hold_time=0.12)
                    menu_escape_cooldown_until = now_loop + 0.8
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
                    attack_key_up()
                    attack_held = False
                release_all_inputs()
                if (now_loop - last_death_respawn_t) >= HP_ZERO_RESPAWN_COOLDOWN_SECONDS:
                    print("[DEATH-HP] HP 0% confirmed -> Respawn + Vote Reset")
                    last_death_respawn_t = now_loop
                    attempt_respawn(sct, full_mon)
                    sleep_interruptible(RESPAWN_TO_RESET_DELAY_SECONDS)
                    auto_esc_ignore_until = time.time() + 1.0
                    if _run_vote_reset(target_zone):
                        _do_zone_reset()
                sleep_interruptible(0.08)
                continue

            if hp is None and now_loop < hp_bar_missing_until:
                sleep_interruptible(0.05)
                continue

            hp_for_potion = hp if hp is not None else last_known_hp
            if hp_for_potion is not None:
                hp_is_zero_like = hp_for_potion <= HP_ZERO_DEAD_THRESHOLD
                if not hp_is_zero_like and hp_for_potion < HP_RECOVERY_THRESHOLD \
                        and (now_loop - last_hp_recovery_t) >= HP_RECOVERY_COOLDOWN_SECONDS:
                    print(f"[HP] {hp_for_potion:.0%} < {HP_RECOVERY_THRESHOLD:.0%} -> potion spam + spin")
                    for _ in range(HP_RECOVERY_POTION_PRESSES):
                        tap_key(HEALTH_POTION_KEY, hold_time=0.06)
                        sleep_interruptible(0.05)
                    last_health_potion_t = now_loop
                    last_hp_recovery_t   = now_loop
                    spin_to_clear_melee()
                    if attack_held:
                        attack_key_down()
                    continue
                potion_window = (
                    HEALTH_POTION_BURST_MIN_HP <= hp_for_potion <= HEALTH_POTION_BURST_MAX_HP
                )
                if potion_window and not hp_is_zero_like and \
                        (now_loop - last_health_potion_t) >= HEALTH_POTION_COOLDOWN_SECONDS:
                    for _ in range(HEALTH_POTION_BURST_COUNT):
                        tap_key(HEALTH_POTION_KEY, hold_time=0.06)
                        time.sleep(HEALTH_POTION_BURST_GAP_SECONDS)
                    last_health_potion_t = now_loop
                _pkt_t = _st.get_tracker()
                _flee_thr = HP_FLEE_THRESHOLD * (
                    0.55 if (_pkt_t and _pkt_t.available
                             and _st.BUFF_CLAIRVOYANCE in _pkt_t.active_buffs) else 1.0
                )
                if (hp or 0.0) < _flee_thr and state != State.FLEE:
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
                dots = [d for d in dots if 0 <= d[0] < w_mm and 0 <= d[1] < h_mm]

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

            if filtered_enemy_dots:
                last_any_enemy_seen_t = now_loop

            n_enemies = len(filtered_enemy_dots)
            if n_enemies != _log_enemy_count:
                _log_enemy_count = n_enemies
                if n_enemies > 0:
                    from detectors import ENEMY_TIER_SKULL, ENEMY_TIER_YELLOW, ENEMY_TIER_PURPLE, ENEMY_TIER_RED
                    _tier_names = {ENEMY_TIER_SKULL: "skull", ENEMY_TIER_YELLOW: "yellow",
                                   ENEMY_TIER_PURPLE: "purple", ENEMY_TIER_RED: "red"}
                    _counts = {}
                    for _, _, t in filtered_enemy_dots:
                        _counts[t] = _counts.get(t, 0) + 1
                    _summary = " ".join(f"{v}x{_tier_names.get(k,'?')}" for k, v in sorted(_counts.items(), reverse=True))
                    print(f"[DETECT] {n_enemies} enem{'y' if n_enemies==1 else 'ies'}: {_summary}")

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
                escape_committed_until = 0.0
            else:
                no_enemy_frames += 1

            reset_enemy_clear = (
                no_enemy_frames >= NO_ENEMY_RESET_MIN_FRAMES
                and (now_loop - last_enemy_seen_t) >= VOTE_RESET_ENEMY_GRACE_SECONDS
                and (now_loop - last_any_enemy_seen_t) >= ENEMY_SOFT_RESET_GRACE_SECONDS
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

            # ── vote-reset triggers ───────────────────────────────────────────
            cooldown_ok = (
                now_loop >= vote_reset_lockout_until
                and (now_loop - last_vote_reset_t) >= VOTE_RESET_COOLDOWN_SECONDS
            )

            if hard_stuck_trigger and confirmed_enemy_now and cooldown_ok:
                print(f"[STUCK-ENEMY] Stuck {hard_stuck_age:.1f}s with visible enemy -> Vote Reset")
                _attempt_vote_reset(with_spin=True)
                continue

            if zone_had_attack and reset_safe_window and cooldown_ok and \
                    (now_loop - last_attack_activity_t) >= ATTACK_IDLE_VOTE_RESET_SECONDS:
                if attack_held:
                    attack_key_up()
                    attack_held = False
                print("No attack for 5s -> Vote Reset")
                _attempt_vote_reset(with_spin=True)
                continue

            if reset_safe_window and cooldown_ok and \
                    (now_loop - zone_started_t) >= ZONE_MAX_SECONDS_BEFORE_RESET:
                if attack_held:
                    attack_key_up()
                    attack_held = False
                print(f"Zone > {ZONE_MAX_SECONDS_BEFORE_RESET:.0f}s without useful combat -> Vote Reset")
                _attempt_vote_reset(with_spin=True)
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
                    _attempt_vote_reset(with_spin=True)
                continue

            if stuck_reset_safe_window and cooldown_ok and \
                    (stuck_no_enemy_count >= STUCK_VOTE_RESET_CONSECUTIVE or hard_stuck_trigger):
                if hard_stuck_trigger and \
                        corner_escape_attempts < CORNER_ESCAPE_MAX_ATTEMPTS_BEFORE_RESET and \
                        (now_loop - last_corner_escape_t) >= CORNER_ESCAPE_COOLDOWN_SECONDS:
                    print(f"Hard-Stuck {hard_stuck_age:.1f}s -> Corner Escape Burst "
                          f"{corner_escape_attempts + 1}/{CORNER_ESCAPE_MAX_ATTEMPTS_BEFORE_RESET}")
                    if attack_held:
                        attack_key_up()
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
                    attack_key_up()
                    attack_held = False
                if hard_stuck_trigger:
                    print(f"Hard-stuck {hard_stuck_age:.1f}s without enemies -> Vote Reset")
                else:
                    print(f"{stuck_no_enemy_count}x stuck without enemies -> Vote Reset")
                _attempt_vote_reset(with_spin=True)
                continue

            # ── right-click shield refresh ────────────────────────────────────
            if _enable_buff_shield and now_loop >= next_attack_rightclick_t and (now_loop - last_jump_t[0]) >= 0.42:
                attack_held              = cast_key_skill_safe(_buff_shield_key, attack_held, resume_left=True)
                rightclick_busy_until    = time.time() + 0.35
                next_attack_rightclick_t = now_loop + _buff_shield_cd

            # ── enemy triggers ATTACK state ───────────────────────────────────
            if state != State.FLEE and confirmed_enemy_now and nearest:
                adx, ady       = dot_to_screen_delta(nearest, center)
                last_flee_dir  = (-adx, -ady)
                last_enemy_seen_t = time.time()
                if state != State.ATTACK:
                    resource_targeter.release()
                    debug_recorder.maybe_record(
                        now=now_loop, event="enter_attack",
                        frame_bgr=frame, minimap_bgr=minimap, center=center,
                        enemy_dots=[(d[0], d[1], d[2], "?") for d in filtered_enemy_dots],
                        item_dots=[], ore_dots=[],
                        companion_dot=companion._candidate if hasattr(companion, "_candidate") else None,
                        nearest_enemy=nearest, resource_target=resource_targeter.active,
                        walkable_mask=cached_mask, enemy_color_mask=None,
                        state_label="EXPLORE->ATTACK",
                        hp_percent=hp, counters_snapshot={},
                    )
                    print("Enemy found -> ATTACK (priority)")
                    _enter_attack_mode(now_loop, adx, ady, hold_attack=True)
                    enemy_chaser.reset()
                    attack_stuck.reset()
                    _aim_cursor_toward_vector(adx, ady)
                    continue

            # ── opening key-3 drip taps ───────────────────────────────────────
            if zone_opening_key3_taps_remaining > 0 and now_loop >= next_opening_key3_t:
                tap_key('3', hold_time=0.12)
                zone_opening_key3_taps_remaining -= 1
                next_opening_key3_t = now_loop + 0.40

            # ── STATE: EXPLORE ────────────────────────────────────────────────
            if state == State.EXPLORE:
                if attack_held:
                    attack_key_up()
                    attack_held = False

                if time.time() < explore_pause_until:
                    sleep_interruptible(0.05)
                    continue

                # Prioritize exploration/combat acquisition right after zone start.
                resource_collect_allowed = (
                    zone_had_attack
                    or (now_loop - zone_started_t) >= RESOURCE_COLLECT_START_DELAY_SECONDS
                )

                # ── minimap-driven resource targeting ─────────────────────
                # Picks a star/pickaxe on the minimap, walks toward it via A*,
                # then hands off to the on-screen handlers below for the final
                # F-press / QTE step.
                if ENABLE_MINIMAP_RESOURCE_TARGETING and (_enable_mining or _enable_loot_items) \
                        and resource_collect_allowed and not confirmed_enemy_now:
                    qte_active = detect_mining_qte_window(frame).get("active", False)
                    if not qte_active:
                        rt_item_dots = classify_item_dots(minimap) if _enable_loot_items else []
                        rt_ore_dots  = classify_ore_dots(minimap)  if _enable_mining     else []
                        rt_target = resource_targeter.tick(
                            now=now_loop,
                            center=center,
                            item_dots=rt_item_dots,
                            ore_dots=rt_ore_dots,
                            close_threshold_px=MINIMAP_RESOURCE_PICKUP_DIST_PX,
                        )
                        if rt_target is not None and \
                                not resource_targeter.is_close(center, MINIMAP_RESOURCE_PICKUP_DIST_PX):
                            rt_waypoint = _a_star_next_waypoint(cached_mask, center, rt_target.pos)
                            if rt_waypoint is not None:
                                rt_dx, rt_dy = dot_to_screen_delta(rt_waypoint, center)
                                wasd_move(rt_dx, rt_dy, random.uniform(0.12, 0.22))
                                sleep_interruptible(0.03)
                                continue
                        # If target is None or already close, fall through to
                        # the existing on-screen handlers below.

                if _enable_mining and resource_collect_allowed and not confirmed_enemy_now:
                    qte = detect_mining_qte_window(frame)
                    if qte.get("active", False):
                        mining_active_until = now_loop + MINING_SESSION_MEMORY_SECONDS
                        if qte.get("in_window", False) and \
                                (now_loop - last_mining_qte_press_t) >= MINING_QTE_PRESS_COOLDOWN_SECONDS:
                            tap_key('f', hold_time=0.04)
                            last_mining_qte_press_t = now_loop
                        sleep_interruptible(0.03)
                        continue

                    mining_should_scan = (
                        (now_loop - last_mining_scan_t) >= MINING_SCAN_INTERVAL_SECONDS
                        or mining_target_screen is None
                    )
                    ores_on_screen = []
                    if mining_should_scan:
                        ores_on_screen = find_ore_nodes_on_screen(frame)
                        last_mining_scan_t = now_loop

                    if ores_on_screen:
                        px, py = SCREEN_W // 2, SCREEN_H // 2
                        mining_target_screen = min(
                            ores_on_screen,
                            key=lambda n: (n[0] - px) * (n[0] - px) + (n[1] - py) * (n[1] - py),
                        )
                        mining_active_until = now_loop + MINING_SESSION_MEMORY_SECONDS

                    if mining_target_screen is not None and now_loop <= mining_active_until:
                        tx, ty = int(mining_target_screen[0]), int(mining_target_screen[1])
                        mdx = tx - (SCREEN_W // 2)
                        mdy = ty - (SCREEN_H // 2)
                        mdist = math.hypot(mdx, mdy)

                        if (now_loop - last_mining_click_t) >= MINING_CLICK_INTERVAL_SECONDS:
                            left_click_at(tx, ty, hold_time=0.05)
                            last_mining_click_t = now_loop

                        if mdist > MINING_APPROACH_MIN_DIST_PX:
                            wasd_move(mdx, mdy, random.uniform(
                                MINING_APPROACH_MOVE_MIN_SECONDS,
                                MINING_APPROACH_MOVE_MAX_SECONDS,
                            ))
                            if mdist >= MINING_APPROACH_JUMP_DIST_PX:
                                maybe_jump_with_movement(mdx, mdy, last_jump_t, force_aim=True)

                        if (now_loop - last_mining_interact_t) >= MINING_INTERACT_INTERVAL_SECONDS:
                            tap_key('f', hold_time=0.04)
                            last_mining_interact_t = now_loop

                        sleep_interruptible(0.04)
                        continue

                    mining_target_screen = None

                # ── LOOT ITEMS (new): star-marker hint + on-screen tag click ───────────
                _loot_t = _st.get_tracker()
                _loot_goblin_active = (
                    _loot_t and _loot_t.available
                    and any(b in _loot_t.active_buffs
                            for b in ("Loot Goblin I", "Loot Goblin II"))
                )
                if (_enable_loot_items or _loot_goblin_active) and resource_collect_allowed and not confirmed_enemy_now:
                    if (now_loop - last_loot_items_scan_t) >= LOOT_ITEMS_SCAN_INTERVAL_SECONDS:
                        items_on_screen = find_item_tags_on_screen(frame)
                        last_loot_items_scan_t = now_loop

                        if items_on_screen:
                            px, py = SCREEN_W // 2, SCREEN_H // 2
                            tx, ty, _ = min(
                                items_on_screen,
                                key=lambda n: (n[0] - px) * (n[0] - px) + (n[1] - py) * (n[1] - py),
                            )
                            if (now_loop - last_loot_items_click_t) >= LOOT_ITEMS_CLICK_INTERVAL_SECONDS:
                                left_click_at(int(tx), int(ty), hold_time=0.05)
                                tap_key('f', hold_time=0.04)
                                last_loot_items_click_t = now_loop
                            sleep_interruptible(0.03)
                            continue

                pt = ctypes.wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                stuck.update(pt.x, pt.y)
                explore_heatmap.update(center, minimap.shape[:2])

                explore_intent_stuck.update(minimap, explore_dx, explore_dy)

                explore_is_stuck = stuck.is_stuck() or explore_intent_stuck.is_stuck()

                # Honour escape commitment — hold escape direction, suppress navigator.
                if now_loop < escape_committed_until:
                    explore_dx, explore_dy = escape_dx, escape_dy
                    explore_is_stuck       = False

                if explore_is_stuck and now_loop >= escape_committed_until:
                    if not explore_stuck_active:
                        explore_stuck_active = True
                        explore_stuck_jump_pref[0] = True
                    now = time.time()
                    if not confirmed_enemy_now and hard_stuck_since == 0.0:
                        hard_stuck_since = now
                    if (now - last_stuck_log_t) > STUCK_LOG_COOLDOWN:
                        last_stuck_log_t = now
                        _esc_ang = math.degrees(math.atan2(explore_dy, explore_dx))
                        print(f"[STUCK] octant-escape was={_esc_ang:.0f}°")
                    if not confirmed_enemy_now and \
                            (now - last_stuck_count_t) >= STUCK_COUNT_EVENT_COOLDOWN_SECONDS:
                        stuck_no_enemy_count += 1
                        last_stuck_count_t    = now
                    backtrack_move(explore_dx, explore_dy, last_backtrack_t)
                    if (now_loop - last_terrain_block_mark_t) >= TERRAIN_BLOCKED_MARK_COOLDOWN_SECONDS:
                        terrain_memory.mark_blocked(center, explore_dx, explore_dy)
                        last_terrain_block_mark_t = now_loop
                    if now_loop >= rightclick_busy_until:
                        jumped, jump_dir = maybe_jump_stuck_sequence(
                            explore_dx, explore_dy,
                            last_jump_t,
                            explore_stuck_jump_seq_idx,
                            explore_stuck_jump_pref,
                        )
                        if jumped and (now - last_stuck_log_t) > STUCK_LOG_COOLDOWN:
                            print(f"[STUCK-JUMP] explore -> {jump_dir}")
                            last_stuck_log_t = now
                    blocked_explore_angles.append(math.atan2(explore_dy, explore_dx))
                    blocked_explore_angles = blocked_explore_angles[-8:]
                    # Octant escape: pick best walkable direction excluding current
                    esc_dx, esc_dy = octant_escape_delta(
                        cached_mask, center, explore_dx, explore_dy,
                        recent_angles=recent_explore_angles)
                    explore_dx, explore_dy = esc_dx, esc_dy
                    escape_dx, escape_dy   = esc_dx, esc_dy
                    escape_committed_until = now_loop + random.uniform(
                        ESCAPE_COMMIT_MIN_SECONDS, ESCAPE_COMMIT_MAX_SECONDS)
                    explore_navigator.reset()
                elif now_loop >= escape_committed_until:
                    hard_stuck_since       = 0.0
                    corner_escape_attempts = 0
                    explore_stuck_active   = False
                    explore_stuck_jump_pref[0] = False

                # Navigator picks direction when not in a committed escape window.
                if now_loop >= escape_committed_until:
                    move = explore_navigator.get_move(minimap, center, cached_mask)
                    if move is None:
                        recent_enemy_valid = (
                            (now_loop - last_seen_enemy_t) <= LAST_SEEN_MONSTER_MEMORY_SECONDS
                        )
                        new_target = None
                        if recent_enemy_valid:
                            _, new_target = choose_recent_enemy_delta(
                                minimap, center,
                                recent_enemy_dot=last_seen_enemy_dot,
                                recent_enemy_vec=last_seen_enemy_vec,
                                terrain_memory=terrain_memory,
                            )
                        if new_target is None:
                            new_target = minimap_explore_target(
                                minimap, center,
                                recent_explore_angles, blocked_explore_angles,
                                heatmap=explore_heatmap,
                                terrain_memory=terrain_memory,
                                sector_tracker=sector_tracker,
                            )
                        if new_target is not None and explore_navigator.plan(
                                minimap, center, new_target, cached_mask):
                            angle = math.atan2(
                                new_target[1] - center[1],
                                new_target[0] - center[0],
                            )
                            recent_explore_angles.append(angle)
                            recent_explore_angles = recent_explore_angles[-8:]
                            sector_tracker.update(angle)
                            last_explore_target = new_target
                            move = explore_navigator.get_move(minimap, center, cached_mask)
                        if move is None:
                            move = random_explore_delta()
                    explore_dx, explore_dy = move

                explore_held_keys = explore_move_start(explore_dx, explore_dy, explore_held_keys)
                if _enable_movement_ability and \
                        (now_loop - last_movement_ability_t) >= _movement_ability_cd and \
                        now_loop >= rightclick_busy_until:
                    cast_movement_ability(explore_dx, explore_dy, key=_movement_ability_key)
                    last_movement_ability_t = now_loop
                _spd_t = _st.get_tracker()
                _move_spd = (0.65 if (_spd_t and _spd_t.available
                                       and _st.BUFF_ZEPHYS_GRACE in _spd_t.active_buffs)
                             else 1.0)
                sleep_interruptible(random.uniform(EXPLORE_MOVE_MIN_SECONDS * _move_spd,
                                                   EXPLORE_MOVE_MAX_SECONDS * _move_spd))
                if now_loop >= rightclick_busy_until:
                    maybe_jump_with_movement(explore_dx, explore_dy, last_jump_t)

            # ── STATE: ATTACK ─────────────────────────────────────────────────
            elif state == State.ATTACK:
                if not attack_held and now_loop >= rightclick_busy_until:
                    attack_key_down()
                    attack_held = True
                elif attack_held and (now_loop - last_attack_key_refresh_t) >= 0.3:
                    attack_key_down()
                    last_attack_key_refresh_t = now_loop

                if nearest and (now_loop - last_attack_sweep_t) >= ATTACK_SWEEP_INTERVAL_SECONDS:
                    perform_attack_sweep(
                        duration=ATTACK_SWEEP_DURATION_SECONDS,
                        radius=ATTACK_SWEEP_RADIUS,
                        move_dx=attack_move_dx,
                        move_dy=attack_move_dy,
                    )
                    last_attack_sweep_t = time.time()

                if _enable_key3_skill and now_loop >= next_attack_key3_t:
                    attack_held        = cast_key3_skill_safe(attack_held, resume_left=True)
                    next_attack_key3_t = now_loop + _key3_cooldown

                attack_stuck.update(center[0], center[1])
                attack_intent_stuck.update(minimap, attack_move_dx, attack_move_dy)

                if attack_stuck.is_stuck() or attack_intent_stuck.is_stuck():
                    if not attack_stuck_active:
                        attack_stuck_active = True
                        attack_stuck_jump_pref[0] = True
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
                    if (now_loop - last_terrain_block_mark_t) >= TERRAIN_BLOCKED_MARK_COOLDOWN_SECONDS:
                        terrain_memory.mark_blocked(center, attack_move_dx, attack_move_dy)
                        last_terrain_block_mark_t = now_loop
                    if now_loop >= rightclick_busy_until:
                        pref_dx = adx if confirmed_enemy_now else attack_move_dx
                        pref_dy = ady if confirmed_enemy_now else attack_move_dy
                        jumped, jump_dir = maybe_jump_stuck_sequence(
                            pref_dx, pref_dy,
                            last_jump_t,
                            attack_stuck_jump_seq_idx,
                            attack_stuck_jump_pref,
                        )
                        if jumped and (now_loop - last_stuck_log_t) > STUCK_LOG_COOLDOWN:
                            print(f"[STUCK-JUMP] attack -> {jump_dir}")
                            last_stuck_log_t = now_loop
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
                    attack_stuck_active    = False
                    attack_stuck_jump_pref[0] = False

                enemy_dist = None
                if confirmed_enemy_now and nearest:
                    last_enemy_seen_t = time.time()
                    adx, ady          = dot_to_screen_delta(nearest, center)
                    enemy_dist        = dot_distance(nearest, center)
                    last_flee_dir     = (-adx, -ady)

                _pkt_t2 = _st.get_tracker()
                _aftermath = (
                    _pkt_t2 and _pkt_t2.available
                    and _st.BUFF_AFTERMATH in _pkt_t2.active_buffs
                )
                _lost_grace = ATTACK_LOST_GRACE * (2.5 if _aftermath else 1.0)
                # Packet-confirmed kill + no visible enemy → skip long visual grace
                if (not confirmed_enemy_now and not _aftermath
                        and _pkt_t2 and _pkt_t2.available
                        and _pkt_t2.last_kill_t > state_start):
                    _lost_grace = min(_lost_grace, 0.4)
                enemy_gone = (time.time() - last_enemy_seen_t) > _lost_grace
                timed_out  = (time.time() - state_start) > ATTACK_TIMEOUT

                if enemy_gone:
                    if attack_held:
                        attack_key_up()
                        attack_held = False
                    print("Enemy dead -> EXPLORE")
                    state = State.EXPLORE
                    explore_pause_until = time.time() + random.uniform(
                        POST_KILL_SCAN_MIN_SECONDS, POST_KILL_SCAN_MAX_SECONDS)
                    state_start = time.time()
                    continue

                if timed_out and confirmed_enemy_now:
                    state_start = time.time()
                elif timed_out and not confirmed_enemy_now:
                    if attack_held:
                        attack_key_up()
                        attack_held = False
                    print("ATTACK timeout without contact -> EXPLORE")
                    state       = State.EXPLORE
                    state_start = time.time()
                    continue

                if not confirmed_enemy_now and \
                        (now_loop - last_enemy_seen_t) >= ATTACK_NO_CONFIRMED_ENEMY_TIMEOUT_SECONDS:
                    if attack_held:
                        attack_key_up()
                        attack_held = False
                    print("ATTACK without confirmed enemy -> EXPLORE")
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
                    # Override linear approach with A* when enemy is far — navigates around walls
                    if confirmed_enemy_now and nearest and enemy_dist > ATTACK_CHASE_MINIMAP_DIST:
                        chase_dx, chase_dy = enemy_chaser.get_chase_delta(
                            minimap, center, nearest, cached_mask)
                        attack_move_dx, attack_move_dy = chase_dx, chase_dy
                        if now_loop - _log_chase_t >= 2.0:
                            _log_chase_t = now_loop
                            _ang = math.degrees(math.atan2(chase_dy, chase_dx))
                            print(f"[PATH] A* chase dist={enemy_dist:.0f}px -> {_ang:.0f}°")
                    last_attack_move_plan_t = time.time()

                wasd_move_and_attack(
                    attack_move_dx, attack_move_dy,
                    duration=random.uniform(0.95, 1.45),
                    aim_dx=adx, aim_dy=ady,
                )
                if _enable_movement_ability and \
                        (now_loop - last_movement_ability_t) >= _movement_ability_cd and \
                        now_loop >= rightclick_busy_until:
                    cast_movement_ability(adx if confirmed_enemy_now else attack_move_dx,
                                          ady if confirmed_enemy_now else attack_move_dy,
                                          key=_movement_ability_key)
                    last_movement_ability_t = now_loop
                if now_loop >= rightclick_busy_until:
                    maybe_jump_with_movement(attack_move_dx, attack_move_dy, last_jump_t)

            # ── STATE: FLEE ───────────────────────────────────────────────────
            elif state == State.FLEE:
                if nearest:
                    adx, ady = dot_to_screen_delta(nearest, center)
                    _aim_cursor_toward_vector(adx, ady)
                    if not attack_held:
                        attack_key_down()
                        attack_held = True
                elif attack_held:
                    attack_key_up()
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
                _pkt_tf = _st.get_tracker()
                _resume_thr = HP_RESUME_THRESHOLD * (
                    0.65 if (_pkt_tf and _pkt_tf.available
                             and _st.BUFF_CLAIRVOYANCE in _pkt_tf.active_buffs) else 1.0
                )
                if hp is None:
                    if (time.time() - state_start) > FLEE_MAX_WAIT:
                        print("HP missing -> EXPLORE")
                        state       = State.EXPLORE
                        state_start = time.time()
                elif hp >= _resume_thr or (time.time() - state_start) > FLEE_MAX_WAIT:
                    print(f"HP {hp:.0%} -> EXPLORE")
                    state       = State.EXPLORE
                    state_start = time.time()

            sleep_interruptible(random.uniform(0.05, 0.12))

    print("Bot finished.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("Bot stopped via Ctrl+C.")
