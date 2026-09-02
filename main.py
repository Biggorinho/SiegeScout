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
    HP_FLEE_THRESHOLD, HP_RESUME_THRESHOLD, HP_FLEE_CONFIRM_SECONDS,
    HP_ZERO_DEAD_THRESHOLD, HP_ZERO_RESPAWN_COOLDOWN_SECONDS,
    HP_BAR_MISSING_GRACE_SECONDS,
    HEALTH_POTION_KEY, HEALTH_POTION_COOLDOWN_SECONDS,
    HEALTH_POTION_BURST_MIN_HP, HEALTH_POTION_BURST_MAX_HP,
    HEALTH_POTION_BURST_COUNT, HEALTH_POTION_BURST_GAP_SECONDS,
    HP_RECOVERY_THRESHOLD,
    HP_RECOVERY_POTION_PRESSES, HP_RECOVERY_COOLDOWN_SECONDS,
    ATTACK_LOST_GRACE, ATTACK_TIMEOUT, FLEE_MAX_WAIT,
    ATTACK_MOVE_REPLAN_SECONDS, ATTACK_CHASE_MINIMAP_DIST,
    MELEE_CHASE_MINIMAP_DIST,
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
    VOTE_RESET_UNAVAILABLE_RETRY_SECONDS, VOTE_RESET_INTERACT_KEY,
    SAFE_ZONE_ENEMY_QUIET_SECONDS,
    VOTE_RESET_FAIL_STREAK_LOCKOUT_COUNT, VOTE_RESET_FAIL_LOCKOUT_SECONDS,
    VOTE_RESET_MIN_HP, NO_ENEMY_RESET_MIN_FRAMES,
    ZONE_MAX_SECONDS_BEFORE_RESET, ZONE_MIN_EXPLORE_BEFORE_RESET_SECONDS,
    ZONE_TRANSITION_FREEZE_SECONDS,
    ZONE_PACKET_MIN_FREEZE_SECONDS,
    ZONE_PACKET_SETTLE_SECONDS,
    DEATH_RESPAWN_COOLDOWN_SECONDS, DEATH_CONFIRM_FRAMES,
    RESPAWN_TO_RESET_DELAY_SECONDS,
    LAST_SEEN_MONSTER_MEMORY_SECONDS,
    COMPANION_TRACK_RADIUS,
    ATTACK_IDLE_VOTE_RESET_SECONDS,
    ESCAPE_COMMIT_MIN_SECONDS, ESCAPE_COMMIT_MAX_SECONDS,
    MINING_SCAN_INTERVAL_SECONDS,
    MINING_CLICK_INTERVAL_SECONDS,
    MINING_INTERACT_INTERVAL_SECONDS,
    MINING_QTE_PRESS_COOLDOWN_SECONDS,
    LOOT_ITEMS_SCAN_INTERVAL_SECONDS,
    LOOT_ITEMS_UNREACHABLE_COOLDOWN_SECONDS,
    LOOT_ITEMS_CLICK_INTERVAL_SECONDS,
    MINING_APPROACH_MIN_DIST_PX,
    MINING_APPROACH_JUMP_DIST_PX,
    MINING_APPROACH_MOVE_MIN_SECONDS,
    MINING_APPROACH_MOVE_MAX_SECONDS,
    MINING_SESSION_MEMORY_SECONDS,
    RESOURCE_COLLECT_START_DELAY_SECONDS,
    ENABLE_MINIMAP_RESOURCE_TARGETING,
    MINIMAP_RESOURCE_PICKUP_DIST_PX,
    RT_STATIC_GIVEUP_SECONDS,
    MINING_UNREACHABLE_COOLDOWN_SECONDS,
    ATTACK_ESCAPE_COMMIT_SECONDS,
    PAUSE_ESC_MAX_ATTEMPTS,
    PAUSE_ESC_LOOP_WINDOW_SECONDS,
    PAUSE_ESC_SUPPRESS_SECONDS,
    MINIMAP_RESOURCE_VISIT_MEMORY_SECONDS,
    MINIMAP_RESOURCE_MAX_TRAVEL_SECONDS,
    MINIMAP_RESOURCE_SPOT_MATCH_RADIUS_PX,
    MINIMAP_RESOURCE_CLOSE_ATTEMPT_SECONDS,
    RESOURCE_PICKUP_SECOND_F_DELAY_SECONDS,
    LOOT_SWEEP_MAX_SECONDS,
    ENABLE_DEBUG_SNAPSHOTS,
    DEBUG_DIR,
    DEBUG_PERIODIC_INTERVAL_SECONDS,
    DEBUG_MAX_FILES,
    DEBUG_EVENT_RATE_LIMIT_SECONDS,
    DEBUG_MOSAIC_PANEL_HEIGHT_PX,
    ENABLE_FRAME_CAPTURE,
    CAPTURE_DIR,
    CAPTURE_INTERVAL_SECONDS,
    CAPTURE_MAX_PAIRS,
    ENEMY_DOT_CONFIRM_THRESHOLD,
    ENEMY_DOT_MATCH_RADIUS_PX,
    ITEM_DOT_CONFIRM_THRESHOLD,
    RESOURCE_TARGET_EARLY_DROP_SECONDS,
    ITEM_DOT_MATCH_RADIUS_PX,
    ORE_DOT_CONFIRM_THRESHOLD,
    ORE_DOT_MATCH_RADIUS_PX,
    DOT_CONFIDENCE_DECAY_PER_MISS,
    DOT_CONFIDENCE_MAX,
)
import config
import fusion
import targeting
from detectors import (
    classify_enemy_dots, get_hp_percent,
    is_death_screen, is_world_map_visible, is_safe_zone_banner_visible,
    is_pause_menu_visible, is_options_menu_visible, is_inventory_panel_visible,
    find_ore_nodes_on_screen, detect_mining_qte_window,
    find_item_tags_on_screen,
    init_healthbar_detector, detect_healthbars,
    init_minimap_vision_detector,
    classify_item_dots, classify_ore_dots,
    build_enemy_color_mask,
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
    minimap_explore_target, resource_approach_delta,
)
from state import State, StuckDetector, IntentStuckDetector, ExploreHeatMap, TerrainMemory, CompanionSuppressor, SectorTracker, EnemyChaser, ExploreNavigator, ResourceTargeter, DetectionCounters, TemporalDotConfidence
from screen_motion import ScreenMotionDetector
from input_control import (
    sleep_interruptible, wasd_move, backtrack_move,
    explore_move_start, explore_move_stop,
    attack_key_down, attack_key_up,
    cast_skill_rule,
    spin_to_clear_melee,
    tap_key, release_all_inputs,
    left_click_at,
    _aim_cursor_toward_vector, maybe_jump_with_movement,
    maybe_jump_stuck_sequence,
    wasd_move_and_attack, perform_attack_sweep,
)
from combat import (
    run_vote_reset_sequence, attempt_respawn, perform_corner_escape_burst,
    pick_zone_from_open_map, wait_for_world_map,
    undo_esc_if_pause_menu_opened, pause_menu_stack_confirmed,
    handle_stray_world_map, safe_zone_walk_delta,
)
from debug_utils import loot_capture_reason, DebugRecorder, FrameCapture
from skills import LoopState, SkillRunner, default_rules, parse_rules

# How long an aim vector stays usable after a state block wrote it. The skill
# block runs before the state machine, so it always reads the previous frame's
# direction; anything older than this means the bot is not moving under its own
# command, and `while_moving` must not fire.
#
# The stamp is real time at the moment the direction was committed, NOT the
# frame's start: the ATTACK block writes the vector *after* wasd_move_and_attack
# blocks for up to 1.45s, so a now_loop stamp would already be ~1.2s stale when
# written and `while_moving` would never hold in ATTACK at all.
AIM_VECTOR_FRESH_SECONDS = 1.0


def run(
    stop_event=None, enable_death_handling=None, enable_mining=None, enable_loot_items=None,
    on_vote_reset=None, target_zone=None,
    enable_rightclick_skill=None, rightclick_cooldown=None,
    enable_key3_skill=None, key3_cooldown=None,
    enable_movement_ability=None, movement_ability_cooldown=None,
    skill_rules=None,
    melee_mode=None,
    enable_debug_snapshots=None,
):
    import config as _cfg
    _enable_death_handling   = _cfg.ENABLE_DEATH_HANDLING if enable_death_handling is None else enable_death_handling
    _melee_mode              = _cfg.MELEE_MODE            if melee_mode is None            else melee_mode
    _enable_mining           = _cfg.ENABLE_MINING         if enable_mining is None         else enable_mining
    _enable_loot_items       = _cfg.ENABLE_LOOT_ITEMS     if enable_loot_items is None     else enable_loot_items
    _enable_debug_snapshots  = _cfg.ENABLE_DEBUG_SNAPSHOTS if enable_debug_snapshots is None else enable_debug_snapshots
    # Skill rules replace the three hardcoded abilities. A caller may pass
    # SkillRule objects or the raw settings rows; anything unusable is
    # dropped rather than taking the bot down with it.
    if skill_rules is None:
        _skill_rules = default_rules()
    elif skill_rules and isinstance(skill_rules[0], dict):
        _skill_rules = parse_rules(skill_rules)
    else:
        _skill_rules = list(skill_rules)
    skill_runner = SkillRunner(_skill_rules)

    print("Bot started | q=Stop | p=Pause | esc=Emergency stop")
    print("Game window must be visible (no background capture with mss)")
    print("Minimap mode: small")
    print("── Settings ──────────────────────────────────")
    print(f"  loot_items     = {_enable_loot_items}")
    print(f"  mining         = {_enable_mining}")
    print(f"  death_handling = {_enable_death_handling}")
    print(f"  melee_mode     = {_melee_mode}")
    print(f"  target_zone    = {target_zone}")
    print(f"  skills         = {len(_skill_rules)} rule(s)")
    for _r in _skill_rules:
        _hp = f" <{_r.hp_percent}%" if _r.trigger == "hp_below" else ""
        _on = "on " if _r.enabled else "off"
        print(f"    [{_on}] {_r.key} {_r.style:6} every {_r.interval:>5.1f}s "
              f"on {_r.trigger}{_hp}")
    print(f"  debug_snapshots= {_enable_debug_snapshots}")
    print("──────────────────────────────────────────────")

    # Initialize Roboflow detectors if enabled
    # - Health bar: Detects enemy health bars on-screen (confirms visible enemies)
    # - Minimap Vision: Detects features directly on the minimap (dungeon, items, portals, etc.)
    init_healthbar_detector()
    init_minimap_vision_detector()

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
    screen_motion                = ScreenMotionDetector(
        window=_cfg.SCREEN_MOTION_WINDOW,
        static_shift_px=_cfg.SCREEN_MOTION_STATIC_PX,
    )
    last_motion_log_t            = 0.0
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
    frame_capture                = FrameCapture(
        enabled=ENABLE_FRAME_CAPTURE,
        base_dir=CAPTURE_DIR,
        interval_seconds=CAPTURE_INTERVAL_SECONDS,
        max_pairs=CAPTURE_MAX_PAIRS,
        loot_enabled=config.CAPTURE_LOOT_FRAMES,
        loot_interval_seconds=config.CAPTURE_LOOT_INTERVAL_SECONDS,
        loot_max=config.CAPTURE_LOOT_MAX,
    )
    last_periodic_debug_t        = 0.0
    loot_kill_capture_until      = 0.0
    counters                     = DetectionCounters(zone_started_at=time.time())
    enemy_confidence             = TemporalDotConfidence(
        match_radius_px=ENEMY_DOT_MATCH_RADIUS_PX,
        confirm_threshold=ENEMY_DOT_CONFIRM_THRESHOLD,
        decay_per_miss=DOT_CONFIDENCE_DECAY_PER_MISS,
        max_confidence=DOT_CONFIDENCE_MAX,
    )
    item_confidence              = TemporalDotConfidence(
        match_radius_px=ITEM_DOT_MATCH_RADIUS_PX,
        confirm_threshold=ITEM_DOT_CONFIRM_THRESHOLD,
        decay_per_miss=DOT_CONFIDENCE_DECAY_PER_MISS,
        max_confidence=DOT_CONFIDENCE_MAX,
    )
    ore_confidence               = TemporalDotConfidence(
        match_radius_px=ORE_DOT_MATCH_RADIUS_PX,
        confirm_threshold=ORE_DOT_CONFIRM_THRESHOLD,
        decay_per_miss=DOT_CONFIDENCE_DECAY_PER_MISS,
        max_confidence=DOT_CONFIDENCE_MAX,
    )
    entered_attack_at            = 0.0
    attack_no_hb_already_fired   = False
    last_qte_in_window_press_t   = 0.0
    enemy_chaser                 = EnemyChaser()
    sector_tracker               = SectorTracker()
    escape_committed_until       = 0.0
    escape_dx, escape_dy         = 0, 0
    terrain_memory               = TerrainMemory()
    attack_stuck                 = StuckDetector(window=ATTACK_STUCK_WINDOW, min_move=ATTACK_STUCK_MIN_MOVE)
    cached_mask                  = None
    companion                    = CompanionSuppressor()
    # Local ML minimap detector (spec §3): None unless enabled AND the model
    # loads. When present, its enemy detections are fused into the heuristic
    # dots each tick; any failure leaves it None and the bot runs as before.
    minimap_ml = None
    if config.LOCAL_MINIMAP_MODEL_ENABLED:
        from local_inference import OnnxDetector
        _mm = OnnxDetector(
            config.LOCAL_MINIMAP_MODEL_PATH,
            config.LOCAL_MINIMAP_MODEL_CLASSES,
            config.LOCAL_MINIMAP_MODEL_INPUT_SIZE,
            conf_threshold=config.LOCAL_MINIMAP_MODEL_CONF,
        )
        minimap_ml = _mm if _mm.available else None
        print(f"[MINIMAP_ML] enabled; detector "
              f"{'ready' if minimap_ml else 'unavailable — heuristics only'}")
    # Local ML on-screen resource detector: replaces the noisy
    # find_ore_nodes_on_screen heuristic for mining when enabled + loaded.
    screen_ml = None
    if config.LOCAL_SCREEN_MODEL_ENABLED:
        from local_inference import OnnxDetector
        _sm = OnnxDetector(
            config.LOCAL_SCREEN_MODEL_PATH,
            config.LOCAL_SCREEN_MODEL_CLASSES,
            config.LOCAL_SCREEN_MODEL_INPUT_SIZE,
            conf_threshold=config.LOCAL_SCREEN_MODEL_CONF,
        )
        screen_ml = _sm if _sm.available else None
        print(f"[SCREEN_ML] enabled; detector "
              f"{'ready' if screen_ml else 'unavailable — heuristics only'}")

    def _item_tags_on_screen(frame_bgr):
        """On-screen loot labels — model first, heuristic only as fallback.

        The screen model has carried an `item` class trained on the user's own
        loot frames since it was added, but was only ever asked about ore, so
        every pickup ran on `find_item_tags_on_screen` — a red/green/cyan blob
        mask that returns a mean of 93 hits per frame (all 300 captured loot
        frames had >= 10). Both pickup sites click the hit nearest screen
        centre, so the bot clicked fire particles and enemy auras while the
        real label sat at the edge of the screen; that is the
        `explore_stall_loot_tag` signal (598 stalled frames against 2 items
        collected in the last session report). Same model-replaces-heuristic
        rule as ore, for the same reason — see fusion.ml_item_tags.
        """
        if screen_ml is not None:
            return fusion.ml_item_tags(
                screen_ml.detect(frame_bgr),
                min_conf=config.LOCAL_SCREEN_MODEL_CONF,
            )
        return find_item_tags_on_screen(frame_bgr)

    minimap_mode                 = "small"
    last_attack_activity_t       = time.time()
    zone_had_attack              = False
    last_vote_reset_t            = 0.0
    vote_reset_fail_streak       = 0
    vote_reset_lockout_until     = 0.0
    auto_esc_ignore_until        = 0.0
    vote_reset_in_progress       = False
    menu_escape_cooldown_until   = 0.0
    pause_menu_frames            = 0
    options_menu_frames          = 0
    inventory_panel_frames       = 0
    safe_zone_banner_frames      = 0
    safe_zone_recover_until      = 0.0
    safe_zone_f_failures         = 0
    MENU_CONFIRM_FRAMES_LOCAL    = 2
    auto_q_ignore_until          = 0.0
    zone_transition_until        = 0.0
    last_health_potion_t         = 0.0
    last_hp_recovery_t           = 0.0
    last_attack_key_refresh_t    = 0.0
    last_known_hp                = None
    last_attack_sweep_t          = time.time()
    skill_runner.defer_all(time.time())
    # The aim vector for `aim` rules. It cannot be computed before the skill
    # block runs -- explore_dx/dy and adx/ady are set inside the state blocks
    # further down -- so each state block writes the direction it just chose
    # and the skill block reads the most recent one. During EXPLORE that
    # direction is held for EXPLORE_MOVE_MIN/MAX_SECONDS anyway, so it is
    # current in practice. The third slot is the timestamp of that write:
    # nothing ever resets this vector, so without it `moving` would latch
    # true after the bot's first step and `while_moving` would degrade into
    # `always` -- firing the movement ability while looting and while stood
    # still.
    aim_vec = [0.0, 0.0, 0.0]
    # Armed at zone start, consumed once by the skill block after the
    # transition freeze. A list so the nested zone-start helper can arm it
    # without a nonlocal, matching last_jump_t / attack_stuck_jump_pref.
    zone_entry_armed = [True]
    hp_zero_since                = 0.0
    hp_low_since                 = 0.0
    hp_bar_missing_until         = 0.0
    stuck_no_enemy_count         = 0
    last_stuck_count_t           = 0.0
    no_enemy_frames              = 0
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
    mining_scan_blocked_until    = 0.0
    loot_scan_blocked_until      = 0.0
    last_mining_click_t          = 0.0
    last_mining_interact_t       = 0.0
    last_mining_qte_press_t      = 0.0
    last_mine_adjacent_log_t     = 0.0
    last_item_nudge_log_t        = 0.0
    last_rt_unrouted_log_t       = 0.0
    last_motion_stuck_log_t      = 0.0
    last_rt_stuck_log_t          = 0.0
    rt_stuck_since               = 0.0
    last_rt_stuck_seen_t         = -1e9
    loot_click_stuck_since       = 0.0
    last_loot_click_seen_t       = -1e9
    attack_escape_until          = 0.0
    attack_escape_delta          = None
    pause_esc_attempts           = 0
    last_pause_esc_t             = -1e9
    pause_esc_suppressed_until   = 0.0
    explore_exit                 = "init"
    last_mining_seen_log_t       = 0.0
    last_qte_log_t               = 0.0
    last_minimap_ore_t           = -1e9
    last_minimap_item_t          = -1e9
    world_map_frames             = 0
    mining_active_until          = 0.0
    mining_target_screen         = None
    last_loot_items_scan_t       = 0.0
    last_loot_items_click_t      = 0.0
    last_resource_log_t          = 0.0   # rate-limit for [LOOT] minimap-dot logging
    loot_sweep_until             = 0.0   # defer vote-reset until this time while looting
    last_terrain_block_mark_t    = 0.0
    _zone_logged_reset_t         = 0.0   # last_vote_reset_t for which zone confirm was printed
    # ── shared reset helper ────────────────────────────────────────────────────
    # Defined as a closure so it can assign back into run()'s local scope.
    def _do_zone_reset():
        nonlocal entered_attack_at, attack_no_hb_already_fired, last_qte_in_window_press_t
        nonlocal last_vote_reset_t, zone_transition_until, last_attack_activity_t
        nonlocal zone_had_attack, stuck_no_enemy_count, hard_stuck_since
        nonlocal corner_escape_attempts, no_enemy_frames, hp_bar_missing_until
        nonlocal zone_started_t, zone_opening_rightclick_pending, state, state_start
        nonlocal explore_pause_until, death_screen_frames, hp_zero_since, hp_low_since
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
        zone_entry_armed[0] = True
        state                         = State.EXPLORE
        state_start                   = last_vote_reset_t
        explore_pause_until           = zone_transition_until
        death_screen_frames           = 0
        hp_zero_since                 = 0.0
        hp_low_since                  = 0.0
        last_any_enemy_seen_t         = 0.0
        explore_dx, explore_dy        = random_explore_delta()
        last_explore_target           = None

        # New zone means new topology; clear cross-zone memory to avoid stale bias.
        recent_explore_angles.clear()
        blocked_explore_angles.clear()
        explore_heatmap.reset()
        terrain_memory.reset()
        sector_tracker.reset()
        screen_motion.reset()  # new zone = scene jump; forget camera-shift history
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
        counters.log_dump(now=time.time())
        counters.reset(now=time.time())
        enemy_confidence.reset()
        item_confidence.reset()
        ore_confidence.reset()
        entered_attack_at = 0.0
        attack_no_hb_already_fired = False
        last_qte_in_window_press_t = 0.0

    def _enter_attack_mode(now_loop, adx=0, ady=0, hold_attack=False):
        nonlocal state, state_start, hard_stuck_since, escape_committed_until
        nonlocal last_attack_sweep_t
        nonlocal attack_move_dx, attack_move_dy, attack_strafe_sign
        nonlocal kite_committed_until, kite_cached_move, last_attack_move_plan_t
        nonlocal zone_opening_rightclick_pending, attack_held
        nonlocal attack_stuck_active, attack_stuck_jump_pref, attack_stuck_jump_seq_idx
        nonlocal attack_escape_until, attack_escape_delta

        state                          = State.ATTACK
        state_start                    = now_loop
        hard_stuck_since               = 0.0
        escape_committed_until         = 0.0
        skill_runner.defer_all(now_loop)
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
        # A commitment from a previous engagement must not steer this one.
        attack_escape_until            = 0.0
        attack_escape_delta            = None
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
            if ok == "unavailable":
                # Disabled Vote Reset button = the game blocks the vote right
                # now: enemies nearby our detection missed, standing in the
                # safe zone (confirmed 2026-07-18: red coverage 0.05 on the
                # safe-zone menu = true disabled skin), or vote cooldown. Not a
                # click failure (no fail streak); back off briefly, then retry.
                vote_reset_fail_streak = 0
                vote_reset_lockout_until = max(vote_reset_lockout_until, attempt_t + VOTE_RESET_UNAVAILABLE_RETRY_SECONDS)
                print(f"[RST] vote reset unavailable (enemies near / safe zone / cooldown) -> retry in {VOTE_RESET_UNAVAILABLE_RETRY_SECONDS:.0f}s")
                return False
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

            # Post-kill loot capture. It sits here rather than with the
            # dot-driven capture below because the kill hands control to
            # EXPLORE with `explore_pause_until` set, and that pause
            # `continue`s before the resource block ever runs -- which is
            # exactly the window the loot label is on screen in. The dot
            # lists are not computed yet at this point, so only the
            # post-kill reason can fire here; save_loot_frame is
            # rate-limited and capped, so overlapping is harmless.
            if config.CAPTURE_LOOT_FRAMES and loot_capture_reason(
                    time.time(), loot_kill_capture_until, (0, 0), [], [],
                    config.CAPTURE_LOOT_NEAR_PX, config.CAPTURE_LOOT_ON_ORE):
                frame_capture.save_loot_frame(time.time(), frame)

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
                # Static gate: a real death overlay freezes the scene. Red-heavy
                # zone palettes (MosArathim Desert) trip is_death_screen on
                # nearly every frame, and momentary HP misreads during combat
                # opened this branch — but the world was visibly moving then.
                if is_death_screen(frame) and screen_motion.is_static():
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
                    if _run_vote_reset(target_zone) is True:
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
                skill_runner.defer_all(time.time())
                # Lock in the minimap-computed direction for 2.5s so the bot physically
                # moves away from the spawn area before the navigator re-evaluates.
                escape_dx, escape_dy       = explore_dx, explore_dy
                escape_committed_until     = time.time() + 2.5
                continue

            # ── stuck-in-menu safety net ──────────────────────────────────────
            # Only fires when no vote-reset is in progress (vote-reset legitimately
            # opens the pause menu). Press ESC to exit Options menu first; on the
            # next iteration the loop will see the pause menu and ESC again.
            # Also skip when HP is critical — death-screen shows red elements
            # that the pause-menu detector false-positives on, and ESC during
            # the death sequence wastes the respawn click.
            if not vote_reset_in_progress and now_loop >= menu_escape_cooldown_until \
                    and not _hp_critical:
                # Require MENU_CONFIRM_FRAMES_LOCAL consecutive detections
                # before firing ESC — single-frame false positives during
                # combat (red effects, post-kill loot bursts) otherwise
                # trigger ESC and break the bot's attack/movement flow.
                if is_options_menu_visible(frame):
                    options_menu_frames += 1
                else:
                    options_menu_frames = 0
                if is_pause_menu_visible(frame):
                    pause_menu_frames += 1
                else:
                    pause_menu_frames = 0
                if is_inventory_panel_visible(frame):
                    inventory_panel_frames += 1
                else:
                    inventory_panel_frames = 0

                # A real pause/options menu freezes the game (screen static); the
                # red beam attacks that trip these detectors happen during active
                # play (screen moving). Only ESC when the world is actually static,
                # so combat beams can't cause a stray ESC that opens the real menu.
                menu_screen_static = screen_motion.is_static()
                if options_menu_frames >= MENU_CONFIRM_FRAMES_LOCAL and menu_screen_static:
                    print("[MENU] options menu detected (paused) — ESC")
                    auto_esc_ignore_until      = time.time() + 1.0
                    tap_key('esc', hold_time=0.12)
                    # Closing Options returns to the pause menu — close that too
                    # instead of leaving it up for a whole extra loop iteration.
                    if undo_esc_if_pause_menu_opened(sct, full_mon):
                        auto_esc_ignore_until = time.time() + 1.0
                    menu_escape_cooldown_until = now_loop + 0.8
                    options_menu_frames = 0
                    continue
                if pause_menu_frames >= MENU_CONFIRM_FRAMES_LOCAL and menu_screen_static \
                        and now_loop >= pause_esc_suppressed_until:
                    # Repeat limiter. Each pass sends ESC, waits, and sends a
                    # second ESC if the menu still looks up — an even number, so
                    # a persistent detection just toggles the menu open/closed
                    # forever. A live run spent ~25s doing exactly that, twice,
                    # both times right after `vote reset unavailable`.
                    # Containment only: WHY the menu survives both ESCs is not
                    # established, and this deliberately does not invent new ESC
                    # behaviour — this is the code path behind the 2026-07-18
                    # Save&Exit incident, where extra menu keypresses exited the
                    # game. Backing off is the safe failure.
                    if (now_loop - last_pause_esc_t) > PAUSE_ESC_LOOP_WINDOW_SECONDS:
                        pause_esc_attempts = 0
                    last_pause_esc_t = now_loop
                    pause_esc_attempts += 1
                    if pause_esc_attempts > PAUSE_ESC_MAX_ATTEMPTS:
                        pause_esc_suppressed_until = now_loop + PAUSE_ESC_SUPPRESS_SECONDS
                        pause_esc_attempts = 0
                        print(f"[MENU] ESC failed to clear the pause menu "
                              f"{PAUSE_ESC_MAX_ATTEMPTS}x -> backing off "
                              f"{PAUSE_ESC_SUPPRESS_SECONDS:.0f}s instead of toggling it")
                        pause_menu_frames = 0
                        continue
                    print("[MENU] pause menu detected (paused) — ESC")
                    auto_esc_ignore_until      = time.time() + 1.0
                    tap_key('esc', hold_time=0.12)
                    # If the detection was a red-zone false positive, that ESC
                    # OPENED the menu (ESC toggles) — verify and toggle it back.
                    if undo_esc_if_pause_menu_opened(sct, full_mon):
                        auto_esc_ignore_until = time.time() + 1.0
                    menu_escape_cooldown_until = now_loop + 0.8
                    pause_menu_frames = 0
                    continue
                # Static gate like the other menu branches: the gold-legend
                # check false-positives on golden zones (Mos'Arathim sand fills
                # the bottom-right band), and its stray ESC opened real menus
                # mid-combat — kicking off the dim -> HP-misread -> phantom
                # death cascade. A real panel means the bot stopped: static.
                if inventory_panel_frames >= MENU_CONFIRM_FRAMES_LOCAL and menu_screen_static:
                    print("[MENU] inventory/interface panel detected — ESC")
                    auto_esc_ignore_until      = time.time() + 1.0
                    tap_key('esc', hold_time=0.12)
                    # 2026-07-18 19:44 incident: this detector false-positived on
                    # MosArathim sand with no panel open, so the ESC OPENED the
                    # pause menu; the dim overlay then made HP read 0%, which
                    # disabled this whole net (_hp_critical gate), and a jump
                    # space-tap activated the hovered Save & Exit. Verify the ESC
                    # result and toggle the menu back closed immediately.
                    if undo_esc_if_pause_menu_opened(sct, full_mon):
                        auto_esc_ignore_until = time.time() + 1.0
                    menu_escape_cooldown_until = now_loop + 0.8
                    inventory_panel_frames = 0
                    continue

            elif not vote_reset_in_progress and now_loop >= menu_escape_cooldown_until:
                # _hp_critical path (the net above is off): HP reads ~0% whenever
                # the pause menu's dim overlay hides the bar — exactly the state
                # where a stray-opened menu most needs closing. On 2026-07-18 the
                # gate kept the menu up until a jump space-tap pressed the hovered
                # Save & Exit and exited to the lobby. The red-skin detectors
                # false-positive on the death screen (why the gate exists), but
                # the clean 4-button stack cannot appear there — trust the stack
                # alone and close the menu.
                if pause_menu_stack_confirmed(frame):
                    print("[MENU] pause menu stack while HP reads critical (dim misread) — ESC")
                    auto_esc_ignore_until      = time.time() + 1.0
                    tap_key('esc', hold_time=0.12)
                    menu_escape_cooldown_until = now_loop + 0.8
                    continue

            # ── safe-zone waypoint recovery ───────────────────────────────────
            # After a vote reset the player is teleported onto the waypoint in
            # the zone's protected entrance (green 'Safe Zone' banner by the
            # minimap). Walking first leaves the waypoint and strands the bot
            # in the safe area — so while the banner is up, press F on the
            # spot and reselect the zone instead of exploring. The zone-start
            # grace keeps a freshly picked zone from being instantly re-reset.
            if not vote_reset_in_progress and now_loop >= safe_zone_recover_until \
                    and (now_loop - zone_started_t) >= 6.0:
                if (now_loop - last_any_enemy_seen_t) < SAFE_ZONE_ENEMY_QUIET_SECONDS:
                    # Enemies on the minimap mean this is not the safe zone,
                    # whatever the banner detector thinks. Zeroing the counter
                    # rather than just skipping the check keeps a stale frame
                    # from firing the moment the last enemy dies.
                    safe_zone_banner_frames = 0
                elif is_safe_zone_banner_visible(frame):
                    safe_zone_banner_frames += 1
                else:
                    safe_zone_banner_frames = 0
                if safe_zone_banner_frames >= 2:
                    safe_zone_banner_frames = 0
                    safe_zone_recover_until = now_loop + 10.0
                    release_all_inputs()
                    attack_held = False
                    print("[SAFE] Safe Zone banner -> pressing F on waypoint to reselect zone")
                    tap_key(VOTE_RESET_INTERACT_KEY, hold_time=0.08)
                    map_visible, map_frame = wait_for_world_map(
                        sct, full_mon, retry_interact_key=VOTE_RESET_INTERACT_KEY)
                    if map_visible and pick_zone_from_open_map(
                            sct, full_mon, map_frame, target_zone=target_zone) is True:
                        safe_zone_f_failures = 0
                        _do_zone_reset()
                    else:
                        # F only works ON the waypoint — after chasing enemies
                        # the player often ends a few steps off it (2026-07-18:
                        # 60+ F presses in place, 10min idle). Walk a rotating
                        # nudge so the next attempt fires from a new spot.
                        wdx, wdy = safe_zone_walk_delta(safe_zone_f_failures)
                        safe_zone_f_failures += 1
                        print(f"[SAFE] F had no effect (x{safe_zone_f_failures}) -> walk nudge ({wdx},{wdy})")
                        wasd_move(wdx, wdy, 0.55)
                    continue

            # ── stray world map net ───────────────────────────────────────────
            # A mis-scrolled/failed zone pick can leave the world map open in
            # gameplay; the HP bar is then unreadable, so the bot fled in place
            # for 50s (the menu net above is skipped when HP reads critical —
            # which the open map itself causes). Independent net, ESC it shut.
            if not vote_reset_in_progress and now_loop >= menu_escape_cooldown_until:
                if is_world_map_visible(frame):
                    world_map_frames += 1
                else:
                    world_map_frames = 0
                if world_map_frames >= 2 and screen_motion.is_static():
                    # An unprompted map is (virtually always) the zone-select
                    # map auto-opened by a passed vote reset — USE it. The old
                    # plain ESC threw away the map 2s after a reset was
                    # mis-reported failed and stranded the bot in the safe
                    # zone for 10min (2026-07-18 22:06). ESC only if no zone
                    # can be picked from it.
                    print("[MENU] world map open during gameplay — trying zone pick")
                    auto_esc_ignore_until = time.time() + 1.0
                    if handle_stray_world_map(sct, full_mon, target_zone=target_zone):
                        print("[MENU] stray map consumed -> zone picked")
                        safe_zone_f_failures = 0
                        _do_zone_reset()
                    menu_escape_cooldown_until = now_loop + 0.8
                    world_map_frames = 0
                    continue

            # ── capture minimap ───────────────────────────────────────────────
            minimap_region = get_minimap_region(minimap_mode)
            minimap        = cv2.cvtColor(np.array(sct.grab(minimap_region)), cv2.COLOR_BGRA2BGR)

            # Camera-shift / "am I physically moving?" — measured off the on-screen
            # world (menus already handled/continued above, so this is gameplay).
            screen_motion.update(frame)
            if _cfg.ENABLE_SCREEN_MOTION_LOG and (now_loop - last_motion_log_t) >= _cfg.SCREEN_MOTION_LOG_INTERVAL:
                _shift = screen_motion.recent_shift()
                print(f"[MOTION] shift={_shift:.2f}px "
                      f"{'static' if screen_motion.is_static() else 'moving'} "
                      f"(state={state.name} exit={explore_exit})")
                last_motion_log_t = now_loop

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
                # Corroborate before respawn-clicking: a real HP-0 death shows
                # the death overlay on a frozen scene within the sustain window.
                # Residual HP misreads (fully dark frame after a failed reset)
                # otherwise fire respawn clicks and an unwanted zone change.
                if (now_loop - last_death_respawn_t) >= HP_ZERO_RESPAWN_COOLDOWN_SECONDS \
                        and is_death_screen(frame) and screen_motion.is_static():
                    print("[DEATH-HP] HP 0% confirmed -> Respawn + Vote Reset")
                    last_death_respawn_t = now_loop
                    attempt_respawn(sct, full_mon)
                    sleep_interruptible(RESPAWN_TO_RESET_DELAY_SECONDS)
                    auto_esc_ignore_until = time.time() + 1.0
                    if _run_vote_reset(target_zone) is True:
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
                    # If still below FLEE threshold after potion spam, force FLEE
                    # rather than continuing the recovery loop — otherwise the bot
                    # potion-spams in place while taking damage.
                    if hp_for_potion < HP_FLEE_THRESHOLD and state != State.FLEE:
                        print(f"[HP] still {hp_for_potion:.0%} after recovery -> FLEE")
                        state       = State.FLEE
                        state_start = time.time()
                        # FLEE never writes the aim vector, so the previous
                        # frame's direction would still read fresh -- and it
                        # points at the enemy being fled, so an `aim` rule
                        # would dash toward it at low HP. Stale the stamp;
                        # the direction itself stays available.
                        aim_vec[2]  = 0.0
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
                # Two guards, because a single frame used to flip the state and
                # a single frame is exactly what a combat effect over the HP bar
                # produces. A 78-minute run logged 265 "HP 0%" against 3 real
                # deaths, and one zone burned 27 MINUTES — 35% of the session —
                # oscillating `HP 0% -> FLEE` / `HP 99% -> EXPLORE` every ~6s.
                # Each flip counted as combat, so the 22s no-combat reset never
                # fired and the bot only escaped by genuinely dying.
                #
                # 1. Ignore near-zero readings entirely. 0% followed by 99% is
                #    not survivable damage, it is a misread; a REAL death is
                #    owned by the hp_zero_sustained path above, which requires
                #    1.5s of sustain plus a death screen on a static scene.
                # 2. Require the low reading to persist. The death path already
                #    corroborates before acting; the flee path guarded nothing,
                #    so the dangerous action was careful and the disruptive one
                #    was not.
                _hp_now = hp if hp is not None else 1.0
                if HP_ZERO_DEAD_THRESHOLD < _hp_now < _flee_thr:
                    if hp_low_since == 0.0:
                        hp_low_since = now_loop
                else:
                    hp_low_since = 0.0

                if hp_low_since > 0.0 \
                        and (now_loop - hp_low_since) >= HP_FLEE_CONFIRM_SECONDS \
                        and state != State.FLEE:
                    print(f"HP {_hp_now:.0%} for "
                          f"{now_loop - hp_low_since:.1f}s -> FLEE")
                    state       = State.FLEE
                    state_start = time.time()
                    aim_vec[2]  = 0.0   # see the FLEE note above

            # ── enemy detection ───────────────────────────────────────────────
            enemy_dots = classify_enemy_dots(minimap)
            _ml_dets = {}          # reused by the resource scan further down
            if minimap_ml is not None:
                # Fuse in enemies the heuristic missed (recall 0.36 -> 0.74 on
                # the eval set). Additive only: never removes a heuristic dot.
                _ml_dets = minimap_ml.detect(minimap)
                enemy_dots = fusion.fuse_enemy_dots(enemy_dots, _ml_dets.get("enemy", []))
            dots       = [(x, y) for x, y, _, _ in enemy_dots]
            center = minimap_center_from_image(minimap)

            # Detect enemies via health bars on-screen (confirms enemy visibility)
            # This is separate from minimap detection - it confirms enemies are visible in-game
            healthbar_detections = detect_healthbars(frame)
            on_screen_enemy_count = len(healthbar_detections) if healthbar_detections else 0

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

            # Confidence-based confirmation (replaces ENEMY_CONFIRM_FRAMES
            # counter) AND target choice, together: `nearest` is picked from
            # the confirmed dots only. Selecting over every raw dot while
            # gating on "did *any* dot confirm?" let a flickering ML phantom
            # -- usually the nearest, so the preferred one -- be charged as
            # the target while a real enemy elsewhere held the gate open.
            _confirmed_enemy_dots, nearest = targeting.confirmed_enemy_target(
                enemy_confidence, filtered_enemy_dots, now_loop, center,
                cached_mask=cached_mask, walkability_fn=ray_walkability_score,
            )
            confirmed_enemy_now = bool(_confirmed_enemy_dots)
            counters.enemy_dot_lost_before_confirm += enemy_confidence.pop_dropped_unconfirmed_count()

            # Build debug overlays once per frame (only when snapshots enabled).
            _dbg_enemy_mask = build_enemy_color_mask(minimap) if _enable_debug_snapshots else None
            _dbg_companion  = companion._candidate if hasattr(companion, "_candidate") else None
            _dbg_item_dots  = classify_item_dots(minimap)  if _enable_debug_snapshots else []
            _dbg_ore_dots   = classify_ore_dots(minimap)   if _enable_debug_snapshots else []

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

            # Loot sweep: when combat has stopped, defer the routine vote-reset
            # triggers (idle + zone-timeout) while the resource targeter is still
            # actively collecting items/ore, bounded by LOOT_SWEEP_MAX_SECONDS. The
            # stuck/death/HP triggers below are NOT deferred — those need a prompt reset.
            loot_in_progress = (
                (_enable_loot_items or _enable_mining)
                and (hp or 0.0) >= VOTE_RESET_MIN_HP
                and resource_targeter.active is not None
            )
            if loot_in_progress and loot_sweep_until == 0.0:
                loot_sweep_until = now_loop + LOOT_SWEEP_MAX_SECONDS
                print("[LOOT] resources present -> sweeping before reset")
            defer_for_loot = loot_in_progress and now_loop < loot_sweep_until
            if loot_sweep_until != 0.0 and not defer_for_loot:
                print("[LOOT] sweep done")
                loot_sweep_until = 0.0

            if zone_had_attack and reset_safe_window and cooldown_ok and not defer_for_loot and \
                    (now_loop - last_attack_activity_t) >= ATTACK_IDLE_VOTE_RESET_SECONDS:
                if attack_held:
                    attack_key_up()
                    attack_held = False
                print(f"No attack for {ATTACK_IDLE_VOTE_RESET_SECONDS:.0f}s -> Vote Reset")
                _attempt_vote_reset(with_spin=True)
                continue

            if reset_safe_window and cooldown_ok and not defer_for_loot and \
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
                    debug_recorder.maybe_record(
                        now=now_loop, event="hard_stuck",
                        frame_bgr=frame, minimap_bgr=minimap, center=center,
                        enemy_dots=filtered_enemy_dots,
                        item_dots=_dbg_item_dots, ore_dots=_dbg_ore_dots,
                        companion_dot=_dbg_companion, nearest_enemy=nearest,
                        resource_target=resource_targeter.active,
                        walkable_mask=cached_mask, enemy_color_mask=_dbg_enemy_mask,
                        state_label="STUCK", hp_percent=hp,
                        counters_snapshot=counters.snapshot(),
                    )
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

            # ── skill rules ───────────────────────────────────────────────────
            # One place where every configured key fires. The conditions that
            # used to be spelled out at three separate sites now live in the
            # rules, and the guards that used to be spelled out here live in
            # the runner.
            _zone_entered = zone_entry_armed[0] and now_loop >= zone_transition_until
            # Is the aim vector still usable? The runner needs this for two
            # things: `while_moving` triggers, and suppressing any `aim`-style
            # rule that has no direction to aim along. Both reach it as
            # LoopState.moving.
            _aim_fresh = ((abs(aim_vec[0]) + abs(aim_vec[1])) > 0
                          and (now_loop - aim_vec[2]) < AIM_VECTOR_FRESH_SECONDS)
            _skill_state = LoopState(
                now=now_loop,
                in_combat=zone_had_attack,
                enemy_visible=confirmed_enemy_now,
                hp=hp,
                zone_entered=_zone_entered,
                moving=_aim_fresh,
                aim_dx=aim_vec[0], aim_dy=aim_vec[1],
                busy_until=rightclick_busy_until,
                last_jump_t=last_jump_t[0],
            )
            for _rule in skill_runner.due(_skill_state):
                attack_held = cast_skill_rule(_rule, attack_held,
                                              aim_vec[0], aim_vec[1])
                if _rule.style != "aim":
                    rightclick_busy_until = time.time() + 0.35
                # Only a rule that actually fired consumes the zone. Clearing
                # the flag unconditionally lost the zone whenever due() had
                # early-returned on busy_until.
                if _rule.trigger == "zone_entry":
                    zone_entry_armed[0] = False

            # Periodic debug snapshot — at this point all detection state is available.
            if _enable_debug_snapshots and (now_loop - last_periodic_debug_t) >= DEBUG_PERIODIC_INTERVAL_SECONDS:
                debug_recorder.force_record(
                    now=now_loop, event="periodic",
                    frame_bgr=frame, minimap_bgr=minimap, center=center,
                    enemy_dots=filtered_enemy_dots,
                    item_dots=_dbg_item_dots, ore_dots=_dbg_ore_dots,
                    companion_dot=_dbg_companion, nearest_enemy=nearest,
                    resource_target=resource_targeter.active,
                    walkable_mask=cached_mask, enemy_color_mask=_dbg_enemy_mask,
                    state_label=state.name if hasattr(state, "name") else str(state),
                    hp_percent=hp, counters_snapshot=counters.snapshot(),
                )
                last_periodic_debug_t = now_loop

            # Raw pair capture for the offline eval harness (no-op unless enabled).
            frame_capture.maybe_save(now_loop, frame, minimap)

            # ── enemy triggers ATTACK state ───────────────────────────────────
            if state != State.FLEE and confirmed_enemy_now and nearest:
                adx, ady       = dot_to_screen_delta(nearest, center)
                last_flee_dir  = (-adx, -ady)
                last_enemy_seen_t = time.time()
                if state != State.ATTACK:
                    resource_targeter.release()
                    loot_sweep_until = 0.0  # restart loot budget after this fight
                    debug_recorder.maybe_record(
                        now=now_loop, event="enter_attack",
                        frame_bgr=frame, minimap_bgr=minimap, center=center,
                        enemy_dots=[(d[0], d[1], d[2], "?") for d in filtered_enemy_dots],
                        item_dots=_dbg_item_dots, ore_dots=_dbg_ore_dots,
                        companion_dot=_dbg_companion,
                        nearest_enemy=nearest, resource_target=resource_targeter.active,
                        walkable_mask=cached_mask, enemy_color_mask=_dbg_enemy_mask,
                        state_label="EXPLORE->ATTACK",
                        hp_percent=hp, counters_snapshot=counters.snapshot(),
                    )
                    print("Enemy found -> ATTACK (priority)")
                    _enter_attack_mode(now_loop, adx, ady, hold_attack=True)
                    entered_attack_at = now_loop
                    attack_no_hb_already_fired = False
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
                # Which branch this iteration exits through. Stays "move" only
                # when control reaches the actual explore movement at the bottom;
                # every early continue overwrites it. Printed by the [MOTION]
                # line so a stall says WHICH branch is eating the time instead of
                # leaving it to be guessed — five hypotheses have already died
                # here (mining latch, post-kill pause, calibration, resource
                # approach, A* failure).
                explore_exit = "move"
                if attack_held:
                    attack_key_up()
                    attack_held = False

                if time.time() < explore_pause_until:
                    sleep_interruptible(0.05)
                    explore_exit = "explore_pause"
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
                        raw_item_dots = classify_item_dots(minimap) if _enable_loot_items else []
                        if not _enable_mining:
                            raw_ore_dots = []
                        elif _ml_dets:
                            # ML ore REPLACES the heuristic rather than adding
                            # to it: on the eval benchmark the heuristic scores
                            # P 0.28 / R 0.32 with 81 false positives against
                            # the model's P 0.93 / R 0.90 with 7. Reuses this
                            # frame's detections — no second inference pass.
                            raw_ore_dots = fusion.ml_ore_dots(_ml_dets)
                        else:
                            raw_ore_dots = classify_ore_dots(minimap)
                        if raw_ore_dots:
                            last_minimap_ore_t = now_loop
                        if raw_item_dots:
                            last_minimap_item_t = now_loop
                        # Capture a full frame for on-screen resource labeling
                        # only when an ore/item dot is close to the player on
                        # the minimap (so the resource is actually on screen).
                        if config.CAPTURE_LOOT_FRAMES and loot_capture_reason(
                                now_loop, loot_kill_capture_until, center,
                                raw_item_dots, raw_ore_dots,
                                config.CAPTURE_LOOT_NEAR_PX,
                                config.CAPTURE_LOOT_ON_ORE):
                            frame_capture.save_loot_frame(now_loop, frame)
                        if (raw_item_dots or raw_ore_dots) and (now_loop - last_resource_log_t) >= 1.0:
                            print(f"[LOOT] minimap: {len(raw_item_dots)} item(s) {len(raw_ore_dots)} ore on map")
                            last_resource_log_t = now_loop
                        raw_item_dots, raw_ore_dots = companion.filter_resources(raw_item_dots, raw_ore_dots)
                        rt_item_dots = item_confidence.update(raw_item_dots, now_loop)
                        rt_ore_dots  = ore_confidence.update(raw_ore_dots,  now_loop)
                        prev_rt_active = resource_targeter.active
                        rt_target = resource_targeter.tick(
                            now=now_loop,
                            center=center,
                            item_dots=rt_item_dots,
                            ore_dots=rt_ore_dots,
                            close_threshold_px=MINIMAP_RESOURCE_PICKUP_DIST_PX,
                        )
                        if prev_rt_active is not None and rt_target is None:
                            reason = resource_targeter.last_release_reason
                            _rt_age = now_loop - prev_rt_active.acquired_at
                            print(f"[RT] {prev_rt_active.kind} target dropped ({reason}) after {_rt_age:.1f}s")
                            if reason == "disappeared" and _rt_age < RESOURCE_TARGET_EARLY_DROP_SECONDS:
                                counters.resource_target_disappeared_early += 1
                            elif reason == "timeout":
                                counters.resource_target_timeout += 1
                            debug_recorder.maybe_record(
                                now=now_loop, event="resource_target_dropped",
                                frame_bgr=frame, minimap_bgr=minimap, center=center,
                                enemy_dots=filtered_enemy_dots,
                                item_dots=raw_item_dots, ore_dots=raw_ore_dots,
                                companion_dot=_dbg_companion, nearest_enemy=nearest,
                                resource_target=prev_rt_active,
                                walkable_mask=cached_mask, enemy_color_mask=_dbg_enemy_mask,
                                state_label=f"RT_DROP({reason})", hp_percent=hp,
                                counters_snapshot=counters.snapshot(),
                            )
                        elif rt_target is not None and prev_rt_active is None:
                            print(f"[RT] {rt_target.kind} target acquired at {rt_target.pos} "
                                  f"dist={dot_distance(rt_target.pos, center):.0f}px")
                            debug_recorder.maybe_record(
                                now=now_loop, event="resource_target_acquired",
                                frame_bgr=frame, minimap_bgr=minimap, center=center,
                                enemy_dots=filtered_enemy_dots,
                                item_dots=raw_item_dots, ore_dots=raw_ore_dots,
                                companion_dot=_dbg_companion, nearest_enemy=nearest,
                                resource_target=rt_target,
                                walkable_mask=cached_mask, enemy_color_mask=_dbg_enemy_mask,
                                state_label=f"RT_PICK({rt_target.kind})", hp_percent=hp,
                                counters_snapshot=counters.snapshot(),
                            )
                        if rt_target is not None and \
                                not resource_targeter.is_close(center, MINIMAP_RESOURCE_PICKUP_DIST_PX):
                            # A* preferred, straight line when it finds no route.
                            # Doing nothing here (the old behaviour) left the bot
                            # standing still for the full 8s travel timeout in
                            # front of an item it had detected correctly, then
                            # re-acquiring it 12s later — with no log line saying
                            # why. The FLEE branch below has always had this same
                            # fallback; only this branch was missing it.
                            rt_dx, rt_dy, rt_routed = resource_approach_delta(
                                cached_mask, center, rt_target.pos)
                            if not rt_routed and (now_loop - last_rt_unrouted_log_t) >= 1.0:
                                last_rt_unrouted_log_t = now_loop
                                print(f"[RT] no A* route to {rt_target.kind} at {rt_target.pos} "
                                      f"(dist={dot_distance(rt_target.pos, center):.0f}px)"
                                      f" -> straight-line approach")
                            if abs(rt_dx) + abs(rt_dy) > 0:
                                # This branch continues, so it bypasses every
                                # stuck detector in EXPLORE. Making it always
                                # produce a move (see above) turned "stand still
                                # doing nothing" into "walk into a wall forever":
                                # a live run held one ore for 7.7s with the
                                # camera dead static and no stuck line reachable.
                                # Jump first — jumping is how the game gets you
                                # over geometry — and give up on the target if
                                # the world still is not scrolling.
                                if _cfg.ENABLE_SCREEN_MOTION_STUCK and screen_motion.is_static():
                                    if (now_loop - last_rt_stuck_log_t) >= 1.0:
                                        last_rt_stuck_log_t = now_loop
                                        print(f"[RT] world static approaching {rt_target.kind} at "
                                              f"{rt_target.pos} -> jump, then drop if still stuck")
                                    maybe_jump_with_movement(rt_dx, rt_dy, last_jump_t,
                                                             force_aim=True)
                                    # A gap means this branch lapsed and the
                                    # recorded start is stale — it only updates
                                    # while the branch runs, and it runs many
                                    # times a second when active. Without this a
                                    # timer started minutes ago is still counting.
                                    if (now_loop - last_rt_stuck_seen_t) > 1.0:
                                        rt_stuck_since = 0.0
                                    last_rt_stuck_seen_t = now_loop
                                    if rt_stuck_since == 0.0:
                                        rt_stuck_since = now_loop
                                    elif (now_loop - rt_stuck_since) >= RT_STATIC_GIVEUP_SECONDS:
                                        print(f"[RT] {rt_target.kind} unreachable after "
                                              f"{now_loop - rt_stuck_since:.1f}s static -> dropping")
                                        # Proven-unreachable: the strongest
                                        # blocked-terrain evidence the bot ever
                                        # gets, and it was being thrown away.
                                        terrain_memory.mark_blocked(center, rt_dx, rt_dy)
                                        # Capture the failure, not the success.
                                        # A session reached 5 of ~59 engaged
                                        # loot targets an hour; these frames are
                                        # the only record of why the other 54
                                        # were abandoned.
                                        debug_recorder.maybe_record(
                                            now=now_loop, event=f"unreachable_{rt_target.kind}",
                                            frame_bgr=frame, minimap_bgr=minimap, center=center,
                                            enemy_dots=[], item_dots=raw_item_dots,
                                            ore_dots=raw_ore_dots,
                                            companion_dot=_dbg_companion, nearest_enemy=None,
                                            resource_target=rt_target,
                                            walkable_mask=cached_mask,
                                            enemy_color_mask=_dbg_enemy_mask,
                                            state_label=f"UNREACHABLE({rt_target.kind})",
                                            hp_percent=hp,
                                            counters_snapshot=counters.snapshot(),
                                        )
                                        resource_targeter.mark_attempted(now_loop)
                                        rt_stuck_since = 0.0
                                        sleep_interruptible(0.03)
                                        explore_exit = "rt_giveup"
                                        continue
                                else:
                                    rt_stuck_since = 0.0
                                wasd_move(rt_dx, rt_dy, random.uniform(0.12, 0.22))
                                sleep_interruptible(0.03)
                                explore_exit = "rt_approach"
                                continue
                        elif rt_target is not None and rt_target.kind == "ore":
                            # Mining: stand next to the ore, press F, wait ~0.5s, press F.
                            if (now_loop - last_mine_adjacent_log_t) >= 1.0:
                                print(f"[MINE] adjacent to ore dot (dist="
                                      f"{dot_distance(rt_target.pos, center):.0f}px) -> F,F")
                                last_mine_adjacent_log_t = now_loop
                            tap_key('f', hold_time=0.04)
                            sleep_interruptible(RESOURCE_PICKUP_SECOND_F_DELAY_SECONDS)
                            tap_key('f', hold_time=0.04)
                            sleep_interruptible(0.05)
                            explore_exit = "ore_adjacent_F"
                            continue
                        elif rt_target is not None:
                            # Item: LOOT by clicking its coloured name/tier label — the
                            # character then auto-walks onto it and picks it up. Don't issue
                            # any other input until the label is gone, or it cancels the walk.
                            tags = _item_tags_on_screen(frame)
                            if tags:
                                px, py = SCREEN_W // 2, SCREEN_H // 2
                                tx, ty, _ = min(tags, key=lambda t: (t[0]-px)**2 + (t[1]-py)**2)
                                print("[LOOT] item label on screen -> click + collect")
                                left_click_at(int(tx), int(ty), hold_time=0.05)
                                _pick_until = time.time() + _cfg.ITEM_PICKUP_WAIT_SECONDS
                                while time.time() < _pick_until:
                                    if stop_event is not None and stop_event.is_set():
                                        break
                                    _pf = cv2.cvtColor(np.array(sct.grab(full_mon)), cv2.COLOR_BGRA2BGR)
                                    if not _item_tags_on_screen(_pf):
                                        break  # label gone = picked up
                                    sleep_interruptible(0.15)
                                resource_targeter.mark_attempted(now_loop)
                                explore_exit = "item_label_click"
                                continue
                            # No label on-screen yet — nudge toward the dot to bring it in view.
                            if (now_loop - last_item_nudge_log_t) >= 1.0:
                                print("[LOOT] close to item dot, no label on screen -> nudging")
                                last_item_nudge_log_t = now_loop
                            idx, idy = dot_to_screen_delta(rt_target.pos, center)
                            if abs(idx) + abs(idy) > 6:
                                wasd_move(idx, idy, 0.08)
                            sleep_interruptible(0.05)
                            explore_exit = "item_nudge"
                            continue
                        # Target None — fall through to the on-screen handlers below.

                if _enable_mining and resource_collect_allowed and not confirmed_enemy_now:
                    qte = detect_mining_qte_window(frame)
                    if qte.get("active", False):
                        if (now_loop - last_qte_log_t) >= 2.0:
                            print(f"[MINE] QTE active (in_window={qte.get('in_window', False)})")
                            last_qte_log_t = now_loop
                        mining_active_until = now_loop + MINING_SESSION_MEMORY_SECONDS
                        if not qte.get("in_window", False) \
                                and (now_loop - last_qte_in_window_press_t) > 1.0:
                            counters.qte_press_missed_window += 1
                            debug_recorder.maybe_record(
                                now=now_loop, event="qte_miss",
                                frame_bgr=frame, minimap_bgr=minimap, center=center,
                                enemy_dots=[], item_dots=[], ore_dots=[],
                                companion_dot=_dbg_companion, nearest_enemy=None,
                                resource_target=resource_targeter.active,
                                walkable_mask=cached_mask, enemy_color_mask=_dbg_enemy_mask,
                                state_label="QTE_MISS", hp_percent=hp,
                                counters_snapshot=counters.snapshot(),
                            )
                            last_qte_in_window_press_t = now_loop
                        if qte.get("in_window", False) and \
                                (now_loop - last_mining_qte_press_t) >= MINING_QTE_PRESS_COOLDOWN_SECONDS:
                            tap_key('f', hold_time=0.04)
                            last_mining_qte_press_t = now_loop
                            last_qte_in_window_press_t = now_loop
                        sleep_interruptible(0.03)
                        explore_exit = "mining_qte"
                        continue

                    # Gate the loose on-screen scan on recent MINIMAP ore
                    # evidence: the screen detector matches any saturated
                    # blob (snow on pines scored 30-45 'ores' per tundra
                    # frame) while the minimap pickaxe detector is precise.
                    # No minimap ore -> no screen scanning, no phantom clicks.
                    minimap_ore_recent = (now_loop - last_minimap_ore_t) <= 8.0
                    if not minimap_ore_recent:
                        mining_target_screen = None
                    mining_should_scan = (
                        minimap_ore_recent
                        and now_loop >= mining_scan_blocked_until
                        and ((now_loop - last_mining_scan_t) >= MINING_SCAN_INTERVAL_SECONDS
                             or mining_target_screen is None)
                    )
                    ores_on_screen = []
                    if mining_should_scan:
                        if screen_ml is not None:
                            # ML on-screen ore: precise vein boxes instead of
                            # the heuristic's 50-90 phantom "nodes". Same
                            # (x, y, kind, score) shape the rest of this block
                            # expects. Fail-soft: {} -> [] -> retry next scan.
                            ores_on_screen = [
                                (int(d["x"]), int(d["y"]), "ore", d["confidence"])
                                for d in screen_ml.detect(frame).get("ore", [])
                            ]
                        else:
                            ores_on_screen = find_ore_nodes_on_screen(frame)
                        last_mining_scan_t = now_loop

                    if ores_on_screen:
                        px, py = SCREEN_W // 2, SCREEN_H // 2
                        mining_target_screen = min(
                            ores_on_screen,
                            key=lambda n: (n[0] - px) * (n[0] - px) + (n[1] - py) * (n[1] - py),
                        )
                        mining_active_until = now_loop + MINING_SESSION_MEMORY_SECONDS
                        if (now_loop - last_mining_seen_log_t) >= 1.0:
                            print(f"[MINE] {len(ores_on_screen)} ore node(s) on screen, "
                                  f"nearest at {mining_target_screen[:2]}")
                            last_mining_seen_log_t = now_loop

                    if mining_target_screen is not None and now_loop <= mining_active_until:
                        tx, ty = int(mining_target_screen[0]), int(mining_target_screen[1])
                        mdx = tx - (SCREEN_W // 2)
                        mdy = ty - (SCREEN_H // 2)
                        mdist = math.hypot(mdx, mdy)

                        if (now_loop - last_mining_click_t) >= MINING_CLICK_INTERVAL_SECONDS:
                            print(f"[MINE] click ore at ({tx},{ty}) dist={mdist:.0f}px")
                            left_click_at(tx, ty, hold_time=0.05)
                            last_mining_click_t = now_loop

                        if mdist > MINING_APPROACH_MIN_DIST_PX:
                            # This is a straight screen-space walk with no wall
                            # avoidance, and the branch continues past EVERY
                            # stuck detector in EXPLORE. So if the world is not
                            # scrolling we are not reaching this ore and never
                            # will — drop it rather than click it forever. A
                            # live run showed six identical clicks at 477px
                            # while the camera sat still.
                            if _cfg.ENABLE_SCREEN_MOTION_STUCK and screen_motion.is_static():
                                print(f"[MINE] world static while approaching ore at "
                                      f"{mdist:.0f}px -> dropping unreachable target")
                                terrain_memory.mark_blocked(center, mdx, mdy)
                                mining_target_screen = None
                                mining_active_until = 0.0
                                # Clearing the target is not enough: the very
                                # next scan finds the SAME ore still on screen
                                # and re-acquires it, so the bot gives up and
                                # un-gives-up forever. A live run did this ten
                                # times in two seconds on one node at 306px.
                                # Suppress the screen scan briefly so the
                                # decision actually sticks.
                                mining_scan_blocked_until = (
                                    now_loop + MINING_UNREACHABLE_COOLDOWN_SECONDS)
                                sleep_interruptible(0.03)
                                explore_exit = "mine_static_drop"
                                continue
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
                        explore_exit = "mine_approach"
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
                    # Gate the on-screen tag scan on recent MINIMAP item
                    # evidence, exactly as the mining scan above is gated on
                    # minimap ore. Without it this fired on screen tags while
                    # the minimap saw zero items, and clicking a label that
                    # never resolves keeps the label on screen — so it clicked
                    # forever. A live run spent 17s static here, and exit=
                    # loot_item_click was on essentially every stalled frame of
                    # the session. Loot Goblin is exempt: its drops are the
                    # whole point and worth scanning for unconditionally.
                    minimap_item_recent = (now_loop - last_minimap_item_t) <= 8.0
                    if not (minimap_item_recent or _loot_goblin_active):
                        items_on_screen = []
                    elif (now_loop >= loot_scan_blocked_until
                          and (now_loop - last_loot_items_scan_t)
                          >= LOOT_ITEMS_SCAN_INTERVAL_SECONDS):
                        items_on_screen = _item_tags_on_screen(frame)
                        last_loot_items_scan_t = now_loop

                        if items_on_screen:
                            px, py = SCREEN_W // 2, SCREEN_H // 2
                            tx, ty, _ = min(
                                items_on_screen,
                                key=lambda n: (n[0] - px) * (n[0] - px) + (n[1] - py) * (n[1] - py),
                            )
                            # Clicking a label makes the character auto-walk to
                            # it, so a label that never resolves keeps itself on
                            # screen and gets clicked forever. Belt to the gate's
                            # braces: if the world is not scrolling we are not
                            # getting there, so stop feeding this branch and let
                            # the normal stuck handling below run.
                            if _cfg.ENABLE_SCREEN_MOTION_STUCK and screen_motion.is_static():
                                # See the rt_approach branch: a gap means this
                                # lapsed and the start is stale. A live run
                                # reported "not reachable after 65.7s static"
                                # because the timer had been sitting unused.
                                if (now_loop - last_loot_click_seen_t) > 1.0:
                                    loot_click_stuck_since = 0.0
                                last_loot_click_seen_t = now_loop
                                if loot_click_stuck_since == 0.0:
                                    loot_click_stuck_since = now_loop
                                elif (now_loop - loot_click_stuck_since) >= RT_STATIC_GIVEUP_SECONDS:
                                    print(f"[LOOT] item tag at ({int(tx)},{int(ty)}) not "
                                          f"reachable after {now_loop - loot_click_stuck_since:.1f}s "
                                          f"static -> ignoring it")
                                    terrain_memory.mark_blocked(
                                        center, int(tx) - SCREEN_W // 2, int(ty) - SCREEN_H // 2)
                                    # The on-screen tag is visible in this frame
                                    # — the whole point is seeing WHY walking to
                                    # it never arrived.
                                    debug_recorder.maybe_record(
                                        now=now_loop, event="unreachable_item_tag",
                                        frame_bgr=frame, minimap_bgr=minimap, center=center,
                                        enemy_dots=[], item_dots=[], ore_dots=[],
                                        companion_dot=_dbg_companion, nearest_enemy=None,
                                        resource_target=resource_targeter.active,
                                        walkable_mask=cached_mask,
                                        enemy_color_mask=_dbg_enemy_mask,
                                        state_label="UNREACHABLE(item_tag)", hp_percent=hp,
                                        counters_snapshot=counters.snapshot(),
                                    )
                                    items_on_screen = []
                                    loot_click_stuck_since = 0.0
                                    last_loot_items_scan_t = now_loop
                                    # Clearing the target is not
                                    # enough: the next scan re-finds
                                    # the same tag a pixel away and
                                    # the give-up loops. Same fix as
                                    # the mining scan's.
                                    loot_scan_blocked_until = (
                                        now_loop
                                        + LOOT_ITEMS_UNREACHABLE_COOLDOWN_SECONDS)
                            else:
                                loot_click_stuck_since = 0.0
                            if (now_loop - last_loot_items_click_t) >= LOOT_ITEMS_CLICK_INTERVAL_SECONDS:
                                left_click_at(int(tx), int(ty), hold_time=0.05)
                                tap_key('f', hold_time=0.04)
                                last_loot_items_click_t = now_loop
                            sleep_interruptible(0.03)
                            explore_exit = "loot_item_click"
                            continue

                pt = ctypes.wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                stuck.update(pt.x, pt.y)
                explore_heatmap.update(center, minimap.shape[:2])

                explore_intent_stuck.update(minimap, explore_dx, explore_dy)

                explore_is_stuck = stuck.is_stuck() or explore_intent_stuck.is_stuck()
                # Camera-shift signal: in EXPLORE the world should scroll; if it's
                # static the bot is walking into geometry. (Loot pickups continue
                # earlier, so this only sees genuine free-exploration stalls.)
                if _cfg.ENABLE_SCREEN_MOTION_STUCK and screen_motion.is_static():
                    # Rate-limited but UNCONDITIONAL. This used to print only on
                    # the transition into stuck, so a stall that was already
                    # flagged went silent — which is why a live run showed 8-14s
                    # of `static (state=EXPLORE)` with no explanation anywhere.
                    if (now_loop - last_motion_stuck_log_t) >= 2.0:
                        last_motion_stuck_log_t = now_loop
                        print(f"[MOTION] world not scrolling "
                              f"({screen_motion.recent_shift():.2f}px) -> stuck"
                              f" (already_stuck={explore_is_stuck})")
                    explore_is_stuck = True

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
                        debug_recorder.maybe_record(
                            now=now, event="stuck_octant",
                            frame_bgr=frame, minimap_bgr=minimap, center=center,
                            enemy_dots=filtered_enemy_dots,
                            item_dots=_dbg_item_dots, ore_dots=_dbg_ore_dots,
                            companion_dot=_dbg_companion, nearest_enemy=nearest,
                            resource_target=resource_targeter.active,
                            walkable_mask=cached_mask, enemy_color_mask=_dbg_enemy_mask,
                            state_label=f"STUCK_OCT({_esc_ang:.0f}°)",
                            hp_percent=hp, counters_snapshot=counters.snapshot(),
                        )
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
                aim_vec[0], aim_vec[1] = explore_dx, explore_dy
                aim_vec[2] = time.time()
                _spd_t = _st.get_tracker()
                _move_spd = (0.65 if (_spd_t and _spd_t.available
                                       and _st.BUFF_ZEPHYS_GRACE in _spd_t.active_buffs)
                             else 1.0)
                sleep_interruptible(random.uniform(EXPLORE_MOVE_MIN_SECONDS * _move_spd,
                                                   EXPLORE_MOVE_MAX_SECONDS * _move_spd))
                if now_loop >= rightclick_busy_until:
                    # Without force_aim the jump only fires when the cursor
                    # already points within ~44° of the walk direction
                    # (JUMP_ALIGNMENT_THRESHOLD 0.72). In EXPLORE the cursor is
                    # parked wherever combat left it, so the jump mostly never
                    # fired — exactly when the bot needs it to clear geometry.
                    # When stuck, aim first so it actually jumps. force_aim
                    # re-aims onto the JUMP_AIM_DISTANCE ring (240px), which is
                    # what keeps a stale cursor off the pause-menu buttons.
                    maybe_jump_with_movement(explore_dx, explore_dy, last_jump_t,
                                             force_aim=explore_is_stuck)

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
                    # Commit to the slide. This block does NOT continue, so
                    # without a commitment the chase below immediately issues
                    # its own 0.95-1.45s move straight back into whatever we are
                    # stuck on — the escape gets 0.18s, the wall-walk gets ~1.2s,
                    # and the bot never frees itself. A live run showed
                    # `dist=98px -> -2° cmd=388px` repeating with the world dead
                    # static and only the 4-5s CHASE TIMEOUT interrupting it.
                    attack_escape_delta = (slide_dx, slide_dy)
                    attack_escape_until = now_loop + ATTACK_ESCAPE_COMMIT_SECONDS
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
                    # Loot is on screen NOW -- open the capture window.
                    loot_kill_capture_until = (
                        time.time() + config.CAPTURE_LOOT_AFTER_KILL_SECONDS)
                    state = State.EXPLORE
                    explore_pause_until = time.time() + random.uniform(
                        POST_KILL_SCAN_MIN_SECONDS, POST_KILL_SCAN_MAX_SECONDS)
                    state_start = time.time()
                    explore_navigator.reset()
                    # Continue exploring in the direction enemies came from —
                    # more enemies are likely deeper in that direction.
                    if last_seen_enemy_vec is not None:
                        _vx, _vy = last_seen_enemy_vec
                        _vlen = math.hypot(_vx, _vy)
                        if _vlen > 1e-6:
                            escape_dx = int((_vx / _vlen) * 200)
                            escape_dy = int((_vy / _vlen) * 200)
                            escape_committed_until = time.time() + 2.5
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
                    explore_navigator.reset()
                    if last_seen_enemy_vec is not None:
                        _vx, _vy = last_seen_enemy_vec
                        _vlen = math.hypot(_vx, _vy)
                        if _vlen > 1e-6:
                            escape_dx = int((_vx / _vlen) * 200)
                            escape_dy = int((_vy / _vlen) * 200)
                            escape_committed_until = time.time() + 2.5
                    continue

                if not confirmed_enemy_now and \
                        (now_loop - last_enemy_seen_t) >= ATTACK_NO_CONFIRMED_ENEMY_TIMEOUT_SECONDS:
                    if attack_held:
                        attack_key_up()
                        attack_held = False
                    print("ATTACK without confirmed enemy -> EXPLORE")
                    state       = State.EXPLORE
                    state_start = time.time()
                    explore_navigator.reset()
                    if last_seen_enemy_vec is not None:
                        _vx, _vy = last_seen_enemy_vec
                        _vlen = math.hypot(_vx, _vy)
                        if _vlen > 1e-6:
                            escape_dx = int((_vx / _vlen) * 200)
                            escape_dy = int((_vy / _vlen) * 200)
                            escape_committed_until = time.time() + 2.5
                    continue

                if (time.time() - last_attack_move_plan_t) > ATTACK_MOVE_REPLAN_SECONDS:
                    (attack_move_dx, attack_move_dy), attack_strafe_sign, \
                    kite_committed_until, kite_cached_move = choose_attack_move_delta(
                        adx, ady, minimap, center, recent_explore_angles,
                        attack_strafe_sign, enemy_dist,
                        kite_committed_until=kite_committed_until,
                        kite_cached_move=kite_cached_move,
                        terrain_memory=terrain_memory,
                        melee_mode=_melee_mode,
                    )
                    # Override linear approach with A* when enemy is far — navigates around walls
                    _chase_thresh = MELEE_CHASE_MINIMAP_DIST if _melee_mode else ATTACK_CHASE_MINIMAP_DIST
                    if now_loop < attack_escape_until and attack_escape_delta is not None:
                        # Escaping takes priority over closing distance; the
                        # chase is what put us against the wall.
                        attack_move_dx, attack_move_dy = attack_escape_delta
                    elif confirmed_enemy_now and nearest and enemy_dist > _chase_thresh:
                        chase_dx, chase_dy = enemy_chaser.get_chase_delta(
                            minimap, center, nearest, cached_mask)
                        attack_move_dx, attack_move_dy = chase_dx, chase_dy
                        if now_loop - _log_chase_t >= 2.0:
                            _log_chase_t = now_loop
                            # Log the MAGNITUDE, not just the heading. atan2(0, 0)
                            # is 0.0, so a stand-still command and "due east" both
                            # printed as 0° — which is exactly what hid the chase
                            # bug fixed in 01c9bc3 for weeks of logs.
                            _mag = math.hypot(chase_dx, chase_dy)
                            if _mag < 1.0:
                                print(f"[PATH] A* chase dist={enemy_dist:.0f}px -> STANDING STILL (cmd 0px)")
                            else:
                                _ang = math.degrees(math.atan2(chase_dy, chase_dx))
                                print(f"[PATH] A* chase dist={enemy_dist:.0f}px -> {_ang:.0f}° cmd={_mag:.0f}px")
                    last_attack_move_plan_t = time.time()

                wasd_move_and_attack(
                    attack_move_dx, attack_move_dy,
                    duration=random.uniform(0.95, 1.45),
                    aim_dx=adx, aim_dy=ady,
                )
                aim_vec[0] = adx if confirmed_enemy_now else attack_move_dx
                aim_vec[1] = ady if confirmed_enemy_now else attack_move_dy
                aim_vec[2] = time.time()
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
