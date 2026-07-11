# ── Display detection + uniform scaling ─────────────────────────────────────
# All pixel coordinates below are calibrated for 1920×1080 windowed-fullscreen.
# At module load we auto-detect the actual primary-monitor resolution and
# rescale every pixel-valued constant uniformly. Works for any 16:9 display
# (1024×576, 1280×720, 1920×1080, 2560×1440, 3840×2160). Non-16:9 monitors
# print a warning — the bot still runs but UI elements may misalign.
import os

_REF_W, _REF_H = 1920, 1080
try:
    import mss as _mss  # mss is already a runtime dependency
    with _mss.mss() as _sct:
        _mon = _sct.monitors[1]  # primary monitor
        SCREEN_W = int(_mon["width"])
        SCREEN_H = int(_mon["height"])
except Exception:
    SCREEN_W, SCREEN_H = _REF_W, _REF_H

_SCALE = SCREEN_W / float(_REF_W)
if abs((SCREEN_W / SCREEN_H) - (_REF_W / _REF_H)) > 0.05:
    # Non-16:9 (e.g. 21:9 ultrawide, 16:10). The game's UI doesn't follow
    # non-standard aspect ratios; the user should pick a 16:9 windowed mode.
    print(f"[CONFIG] WARNING: display {SCREEN_W}x{SCREEN_H} is not 16:9. "
          f"Bot is calibrated for 16:9 — UI elements may misalign. "
          f"Set Hero Siege to a 16:9 windowed-fullscreen mode for best results.")


def _px(value):
    """Scale a 1920×1080-reference pixel value to the current resolution."""
    return int(value * _SCALE)


# Roboflow API Key (shared across all models). Kept out of source control:
# set the ROBOFLOW_API_KEY environment variable, or drop the key into a
# git-ignored roboflow_key.txt next to this file.
def _load_roboflow_key():
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(_here, "roboflow_key.txt"), "r", encoding="utf-8") as _fh:
            return _fh.read().strip()
    except OSError:
        return ""


ROBOFLOW_API_KEY = _load_roboflow_key()

# Roboflow Health Bar Detection (on-screen game UI)
# DISABLED: inference-sdk requires Python 3.9-3.13, but you're running Python 3.14
ROBOFLOW_HEALTHBAR_ENABLED = False  # Detects health bars above enemy heads in-game
ROBOFLOW_HEALTHBAR_CLASSES = ["healthbars"]

# Roboflow Enemy & Dungeon Detection (on-screen game)
ROBOFLOW_ENEMY_DUNGEON_ENABLED = False  # Detects enemy/teleport on main game screen
ROBOFLOW_ENEMY_DUNGEON_CLASSES = ["enemy", "teleport"]

# Roboflow Minimap Vision (on minimap specifically)
ROBOFLOW_MINIMAP_VISION_ENABLED = False  # Detects enemies/items/portals on the minimap
ROBOFLOW_MINIMAP_VISION_CLASSES = ["player"]  # Can also use: dungeon, enter, exit, portal, ore*

MINIMAP_SMALL = {
    "left":   _px(1673),
    "top":    _px(7),
    "width":  _px(225),
    "height": _px(216),
}
MINIMAP_LARGE = {
    "left":   max(0, SCREEN_W // 2 - _px(430)),
    "top":    max(0, SCREEN_H // 2 - _px(350)),
    "width":  _px(860),
    "height": _px(700),
}
MINIMAP_SMALL_NUDGE = {"left": _px(8), "top": _px(-4), "right": _px(18), "bottom": _px(10)}
MINIMAP_LARGE_NUDGE = {"left": 0, "top": 0, "right": 0, "bottom": 0}
HP_BAR = {"x": _px(113), "y": _px(36), "w": _px(173), "h": _px(6)}
HP_FLEE_THRESHOLD            = 0.30
HP_RESUME_THRESHOLD          = 0.50
HP_ZERO_DEAD_THRESHOLD       = 0.02
HP_ZERO_RESPAWN_COOLDOWN_SECONDS = 3.0
HP_BAR_MISSING_GRACE_SECONDS = 3.5
HEALTH_POTION_KEY            = '1'
HEALTH_POTION_COOLDOWN_SECONDS = 4.0
HEALTH_POTION_BURST_MIN_HP   = 0.10
HEALTH_POTION_BURST_MAX_HP   = 0.70
HEALTH_POTION_BURST_COUNT    = 10
HEALTH_POTION_BURST_GAP_SECONDS = 0.06
HP_RECOVERY_THRESHOLD        = 0.50
HP_RECOVERY_POTION_PRESSES   = 15
HP_RECOVERY_COOLDOWN_SECONDS = 2.5

ATTACK_LOST_GRACE = 1.2
PLAYER_EXCLUSION_RADIUS      = _px(14)
PLAYER_EXCLUSION_RADIUS_FRAC = 0.06
MOVE_AXIS_DEADZONE           = 10  # movement-vector threshold; abstract, not screen pixels

EXPLORE_REPLAN_SECONDS       = 10.0
EXPLORE_MOVE_MIN_SECONDS     = 0.15
EXPLORE_MOVE_MAX_SECONDS     = 0.25
POST_KILL_SCAN_MIN_SECONDS   = 0.40
POST_KILL_SCAN_MAX_SECONDS   = 0.80
STUCK_LOG_COOLDOWN           = 2.5

ATTACK_MOVE_REPLAN_SECONDS          = 0.25
ATTACK_KITE_CHANCE                  = 0.85
ATTACK_CHASE_MINIMAP_DIST           = _px(32)
ATTACK_FLEE_MINIMAP_DIST            = _px(18)
ATTACK_BACKPEDAL_STRENGTH           = 0.80
ATTACK_ORBIT_BACKPEDAL_STRENGTH     = 0.28
ATTACK_RIGHTCLICK_INTERVAL_SECONDS  = 10.0
ATTACK_KEY3_INTERVAL_SECONDS        = 50.0
ATTACK_SWEEP_INTERVAL_SECONDS       = 6.0
ATTACK_SWEEP_DURATION_SECONDS       = 1.5
ATTACK_SWEEP_RADIUS                 = _px(210)
ATTACK_NO_CONFIRMED_ENEMY_TIMEOUT_SECONDS = 1.0

KEY3_PRESS_HOLD_SECONDS      = 0.09
KEY3_REPEAT_COUNT            = 2
KEY3_DOUBLE_TAP_GAP_SECONDS  = 0.07

ENEMY_CONFIRM_FRAMES         = 2
ENEMY_LOST_FRAMES            = 3

LAST_SEEN_MONSTER_MEMORY_SECONDS      = 22.0
LAST_SEEN_MONSTER_MIN_DIST            = _px(16)
LAST_SEEN_MONSTER_FALLBACK_DIST_MIN   = _px(90)
LAST_SEEN_MONSTER_FALLBACK_DIST_MAX   = _px(200)

COMPANION_TRACK_RADIUS    = _px(42)
COMPANION_CONFIRM_FRAMES  = 3
COMPANION_SUPPRESS_RADIUS = _px(12)
COMPANION_REACQUIRE_RADIUS = _px(18)
COMPANION_FORGET_SECONDS  = 2.5
COMPANION_CENTER_MAX_DIST = _px(38)
COMPANION_WHITE_SUPPRESS_RADIUS = _px(48)
COMPANION_WHITE_MIN_V           = 200
COMPANION_WHITE_MAX_S           = 72

JUMP_COOLDOWN              = 1.5
JUMP_ALIGNMENT_THRESHOLD   = 0.72
JUMP_AIM_DISTANCE          = _px(240)
AIM_MIN_DISTANCE           = _px(180)
AIM_MAX_DISTANCE           = _px(280)

PATH_GRID_STEP             = 3   # downsampling factor — abstract
PATH_WAYPOINT_STEPS        = 7   # waypoint count — abstract
PATH_WAYPOINT_REACH_PX     = _px(8)
WALL_BUFFER_ITERATIONS     = 2   # erosion iterations — abstract

# Dark-zone walkable model. Some zones (Satanic/Hell dimensions) render the
# walkable floor as near-black, which collapses the bright-floor walkable
# heuristic to a few pixels and leaves the bot unable to path. When the minimap
# median brightness falls below DARK_MINIMAP_MEDIAN_V we switch to a disc model:
# treat the whole minimap circle as floor, minus genuinely bright structures and
# the cyan boundary ring.
DARK_MINIMAP_MEDIAN_V      = 20    # median V below this → dark-zone model
DARK_DISC_RADIUS_FRAC      = 0.95  # fraction of the half-min-dimension used as the disc
DARK_WALL_MIN_V            = 140   # V at/above this (low sat) counts as a wall structure

# Screen-motion (camera-shift) detection — "is the player physically moving?"
# Measures how far the centre-screen world scrolls between frames. Logging is on
# so the real moving-vs-stuck values can be read from a run; the stuck hook stays
# OFF until SCREEN_MOTION_STATIC_PX is calibrated to those values.
ENABLE_SCREEN_MOTION_LOG   = True   # print rate-limited [MOTION] shift values
ENABLE_SCREEN_MOTION_STUCK = True   # use camera-static as an explore stuck signal
SCREEN_MOTION_WINDOW       = 6
SCREEN_MOTION_STATIC_PX    = 1.2    # calibrated: static ~0.4-0.95px, moving >=1.36px
SCREEN_MOTION_LOG_INTERVAL = 2.0

TERRAIN_BLOCKED_PATCH_RADIUS           = _px(14)
TERRAIN_BLOCKED_MARK_AHEAD_PX          = _px(12)
TERRAIN_BLOCKED_MARK_STRENGTH          = 90
TERRAIN_BLOCKED_THRESHOLD              = 42
TERRAIN_BLOCKED_DECAY_EVERY_UPDATES    = 80
TERRAIN_BLOCKED_DECAY_VALUE            = 5
TERRAIN_BLOCKED_MARK_COOLDOWN_SECONDS  = 0.55

STUCK_BACKTRACK_DURATION_MIN  = 0.25
STUCK_BACKTRACK_DURATION_MAX  = 0.45
STUCK_BACKTRACK_COOLDOWN      = 0.60

EXPLORE_HEATMAP_CELL_SIZE  = _px(8)
EXPLORE_HEATMAP_PENALTY    = 30.0

# New — escape commitment
ESCAPE_COMMIT_MIN_SECONDS  = 2.0
ESCAPE_COMMIT_MAX_SECONDS  = 3.5

# Explore stuck detection — window sized for short ticks (~0.2s each → ~5s before firing)
EXPLORE_STUCK_WINDOW        = 25
EXPLORE_STUCK_MIN_MOVE      = _px(8)
# Walk-score mid-cycle replan cooldown — don't redirect more than once per N seconds
WALK_SCORE_REPLAN_COOLDOWN  = 2.0

# New — sector tracker
SECTOR_TRACKER_DECAY_EVERY = 45
SECTOR_TRACKER_PENALTY     = 30.0

ENEMY_CHASE_REPLAN_SECONDS   = 0.25
ENEMY_CHASE_MOVE_THRESHOLD   = _px(6)
ENEMY_CHASE_PROGRESS_MIN_PX  = _px(3)
ENEMY_CHASE_NO_PROGRESS_MAX  = 4   # frame count — abstract

KITE_COMMIT_MIN_SECONDS    = 0.50
KITE_COMMIT_MAX_SECONDS    = 0.80
ATTACK_STUCK_WINDOW        = 8
ATTACK_STUCK_MIN_MOVE      = _px(8)
ATTACK_STUCK_ARM_SECONDS   = 9.0

ITEM_MATCH_THRESHOLD = 0.78

# Minimap item-star detection. Rarity recolours the same star glyph, so the
# detector is colour-agnostic (any saturated bright hue) and separates the star
# from look-alikes by shape + a teleport-cyan colour reject:
#   round NPC/objective dots → high circularity/solidity
#   ore pickaxes             → high solidity
#   grey/thin map structures → low saturation / low extent
#   teleport crosses         → bright cyan
# Tuned for recall first (missing real stars was the main complaint): catch every
# rarity colour and thin star renders, reject the high-confidence non-items.
# Ore pickaxes can still slip through (near-identical at icon size) — ore has its
# own detector/toggle and the resource targeter's visit-memory absorbs the rest.
ITEM_STAR_MIN_SAT           = 80    # reject grey/white structure fragments
ITEM_STAR_MIN_VAL           = 120
ITEM_STAR_MAX_CIRCULARITY   = 0.45  # reject round NPC/objective dots
ITEM_STAR_MAX_SOLIDITY      = 0.70  # reject filled circles
ITEM_STAR_MIN_EXTENT        = 0.26  # reject the thinnest map slivers (keeps thin stars)
ITEM_STAR_MIN_ASPECT        = 0.5
ITEM_STAR_MAX_ASPECT        = 2.0
ITEM_STAR_TELEPORT_CYAN_MAX = 0.25  # patch cyan fraction above this = teleport

# After clicking a dropped item's label, the character auto-walks to it and
# picks it up; wait this long (no other input) for that to finish.
ITEM_PICKUP_WAIT_SECONDS = 3.0

# Minimap ore pickaxes are orange/gold. The old detector matched ANY saturated
# colour and leaned on shape alone, so it tagged red enemy dots, the orange
# companion duck, and even item stars as "ore". Restricting to the orange hue
# removes those false positives while keeping real ore.
ORE_DOT_HUE_LO  = 10
ORE_DOT_HUE_HI  = 26
ORE_DOT_MIN_SAT = 100
ORE_DOT_MIN_VAL = 100

ENABLE_MINING                         = False
ENABLE_LOOT_ITEMS                     = False
MINING_SCAN_INTERVAL_SECONDS          = 0.10
MINING_CLICK_INTERVAL_SECONDS         = 0.42
MINING_INTERACT_INTERVAL_SECONDS      = 0.22
MINING_QTE_PRESS_COOLDOWN_SECONDS     = 0.16
MINING_APPROACH_MIN_DIST_PX           = _px(92)
MINING_APPROACH_JUMP_DIST_PX          = _px(250)
MINING_APPROACH_MOVE_MIN_SECONDS      = 0.12
MINING_APPROACH_MOVE_MAX_SECONDS      = 0.22
MINING_SESSION_MEMORY_SECONDS         = 2.6
RESOURCE_COLLECT_START_DELAY_SECONDS  = 18.0

LOOT_ITEMS_SCAN_INTERVAL_SECONDS      = 0.10
LOOT_ITEMS_CLICK_INTERVAL_SECONDS     = 0.35

# ── minimap-driven resource targeting ────────────────────────────────────
ENABLE_MINIMAP_RESOURCE_TARGETING       = True
MINIMAP_RESOURCE_PICKUP_DIST_PX         = _px(18)   # close enough on minimap to hand off
MINIMAP_RESOURCE_VISIT_MEMORY_SECONDS   = 12.0      # don't revisit failed spots
MINIMAP_RESOURCE_MAX_TRAVEL_SECONDS     = 8.0       # give up after Xs on one target
MINIMAP_RESOURCE_SPOT_MATCH_RADIUS_PX   = _px(8)    # tolerance for "same spot"
MINIMAP_RESOURCE_CLOSE_ATTEMPT_SECONDS  = 4.0       # if close this long, auto-mark attempted
# Once standing on/near a minimap-targeted resource, items and ore are both
# collected by pressing F; ore needs a second F this many seconds later to mine.
RESOURCE_PICKUP_SECOND_F_DELAY_SECONDS  = 0.5
# Loot sweep: when combat ends, defer the vote-reset for up to this long while the
# resource targeter is still actively collecting items/ore from the minimap.
LOOT_SWEEP_MAX_SECONDS                  = 12.0

# ── debug snapshots + detection telemetry ────────────────────────────────
ENABLE_DEBUG_SNAPSHOTS              = False
DEBUG_DIR                           = "debug"
DEBUG_PERIODIC_INTERVAL_SECONDS     = 5.0
DEBUG_MAX_FILES                     = 500
DEBUG_EVENT_RATE_LIMIT_SECONDS      = 1.5
DEBUG_MOSAIC_PANEL_HEIGHT_PX        = 360

# ── Raw frame capture for the offline eval harness (dev tooling) ─────────────
# Saves unannotated frame+minimap pairs during runs so they can be labeled
# and scored (see eval/). Capped by stopping (not rotating) to preserve
# early-session zone diversity.
ENABLE_FRAME_CAPTURE                = True
CAPTURE_DIR                         = os.path.join("eval", "captures")
CAPTURE_INTERVAL_SECONDS            = 2.0
CAPTURE_MAX_PAIRS                   = 400

ENEMY_DOT_CONFIRM_THRESHOLD         = 2
ENEMY_DOT_MATCH_RADIUS_PX           = _px(6)
ITEM_DOT_CONFIRM_THRESHOLD          = 2
ITEM_DOT_MATCH_RADIUS_PX            = _px(8)
ORE_DOT_CONFIRM_THRESHOLD           = 2
ORE_DOT_MATCH_RADIUS_PX             = _px(8)
DOT_CONFIDENCE_DECAY_PER_MISS       = 1
DOT_CONFIDENCE_MAX                  = 8

ATTACK_TIMEOUT = 10.0
FLEE_MAX_WAIT  = 15.0

# The game locks Vote Reset for ~10s after the player deals or takes a hit.
# The pre-reset spin + menu open add ~4-5s after this idle threshold fires,
# so 7s idle puts the button check at ~11-12s past the last hit — just
# outside the lock. (Was 3.0: the check landed at ~7.5s, inside the lock,
# so most attempts clicked a disabled button.)
ATTACK_IDLE_VOTE_RESET_SECONDS          = 7.0
STUCK_VOTE_RESET_CONSECUTIVE            = 3
STUCK_COUNT_EVENT_COOLDOWN_SECONDS      = 1.2
HARD_STUCK_RESET_SECONDS               = 20.0
CORNER_ESCAPE_COOLDOWN_SECONDS         = 1.4
CORNER_ESCAPE_BURST_STEP_SECONDS       = 0.14
CORNER_ESCAPE_MAX_ATTEMPTS_BEFORE_RESET = 3
ZONE_MAX_SECONDS_BEFORE_RESET          = 22.0
ZONE_MIN_EXPLORE_BEFORE_RESET_SECONDS  = 8.0
VOTE_RESET_COOLDOWN_SECONDS            = 25.0
VOTE_RESET_FAIL_COOLDOWN_SECONDS       = 8.0
# Backoff when the game refuses the reset: the Vote Reset button renders a
# dark disabled skin while enemies are nearby (normally clears within ~5s).
# The bot's own detection missed those enemies, so back off briefly and go
# find them. Distinct from click failures: no lockout streak.
VOTE_RESET_UNAVAILABLE_RETRY_SECONDS   = 10.0
VOTE_RESET_FAIL_STREAK_LOCKOUT_COUNT   = 4
VOTE_RESET_FAIL_LOCKOUT_SECONDS        = 75.0
VOTE_RESET_ENEMY_GRACE_SECONDS         = 2.0
NO_ENEMY_RESET_MIN_FRAMES              = 6
VOTE_RESET_MIN_HP                      = 0.80

VOTE_RESET_CLICK_Y_SWEEP_FRAC    = 0.015
VOTE_RESET_FIXED_CLICK_X_PX      = _px(957)
VOTE_RESET_FIXED_CLICK_Y_PX      = _px(485)
VOTE_RESET_INTERACT_KEY          = 'f'
VOTE_RESET_MENU_SETTLE_SECONDS   = 0.40
VOTE_RESET_CLICK_RETRY_DELAY     = 0.60
VOTE_RESET_CLICK_ATTEMPTS        = 2
VOTE_RESET_SHORTCUT_KEY            = 'r'
VOTE_RESET_SHORTCUT_WAIT_SECONDS   = 3.0
SATANIC_NODE_SCAN_ATTEMPTS       = 5
SATANIC_NODE_SCAN_RETRY_SECONDS   = 0.20

MAP_DETECT_TIMEOUT_SECONDS  = 3.2
MAP_DETECT_POLL_SECONDS     = 0.18
MAP_INTERACT_RETRY_SECONDS  = 0.80
MAP_OPEN_DELAY_SECONDS      = 0.55
MAP_SELECT_SETTLE_SECONDS   = 0.70
MAP_NODE_CLICK_Y_OFFSET_FRAC = 0.006

FALLBACK_NODE_X_FRAC = 0.36
FALLBACK_NODE_Y_FRAC = 0.435

SATANIC_NODE_CLICK_RING_RADIUS_PX  = _px(16)
SATANIC_NODE_CLICK_X_OFFSET        = _px(-2)
SATANIC_NODE_CLICK_Y_OFFSET        = _px(16)
# All fixed-node Y coords are calibrated ~27px too low vs actual screen position.
# Applied only in the CV detection click path (not fixed-zone direct click).
SATANIC_NODE_CLICK_Y_CORRECTION    = _px(-27)
SATANIC_NODE_MIN_SCORE            = 16.0  # detection score threshold — abstract

# Fixed world-map node coordinates (screen pixels on 1920x1080 map capture).
# Satanic zone can spawn on any of these nodes.
_SATANIC_FIXED_NODES_REF = {
    # Reference coordinates calibrated on a 1920×1080 world map screenshot.
    "Arctic Tundra": (306, 217),
    "Chilling Lake": (196, 267),
    "Corrupted Cave": (1731, 561),
    "Corrupted Oasis": (820, 219),
    "Crystal Village": (334, 306),
    "Deep Space": (1392, 253),
    "Dry Hills": (764, 354),
    "Event Horizon": (1446, 369),
    "Fields of Battle": (353, 588),
    "Fuji Coast": (1117, 704),
    "Highland Graveyard": (1062, 403),
    "King's Garden": (432, 412),
    "Misty Swamp": (1442, 535),
    "MosArathim Desert": (684, 464),
    "Mt. Fuji": (843, 789),
    "Old Mining Village": (1459, 527),
    "Outskirts of Inoya": (370, 669),
    "Parallel Dimension": (1300, 535),
    "Prison Dungeon": (1011, 270),
    "Pyramid Level 1": (860, 515),
    "Pyramid Level 2": (944, 483),
    "Sea of Karponia": (1248, 782),
    "Snowy Mountains": (418, 153),
    "Steam Train": (1157, 183),
    "Subconscious Mind": (1469, 640),
    "Temple of Zamjo": (1413, 805),
    "The Black Hole": (1491, 459),
    "The Cathedral": (1141, 317),
    "The Depths of Hell": (1249, 437),
    "The Devil's Breach": (698, 679),
    "The Glacial Trail": (467, 265),
    "The Highland Mines": (440, 765),
    "The Nightmare": (578, 730),
    "The Pumpkin Patch": (244, 525),
    "Woodhill Plains": (339, 463),
}
SATANIC_FIXED_NODES = {
    name: (_px(x), _px(y)) for name, (x, y) in _SATANIC_FIXED_NODES_REF.items()
}

ZONE_TRANSITION_FREEZE_SECONDS    = 3.0
# Packet-driven early exit: zone packet shortens the freeze window once
# MIN_FREEZE seconds have elapsed and SETTLE seconds have passed since the packet.
ZONE_PACKET_MIN_FREEZE_SECONDS    = 1.5
ZONE_PACKET_SETTLE_SECONDS        = 0.5

DEATH_RESPAWN_X_FRAC              = 0.50
DEATH_RESPAWN_Y_FRAC              = 0.83
DEATH_RESPAWN_COOLDOWN_SECONDS    = 3.0
DEATH_RESPAWN_CLICK_ATTEMPTS      = 5
DEATH_RESPAWN_RING_RADIUS_PX      = _px(18)
DEATH_RESPAWN_POST_CLICK_CHECK_SECONDS = 0.18
DEATH_CONFIRM_FRAMES              = 3
ENABLE_DEATH_HANDLING             = True
RESPAWN_TO_RESET_DELAY_SECONDS    = 1.1

ENABLE_LOOT = False

# ── Skill / Ability toggles & cooldowns ──────────────────────────────────────
ENABLE_BUFF_SHIELD_SKILL           = True
BUFF_SHIELD_KEY                    = 'e'
ENABLE_KEY3_SKILL                  = True
ENABLE_MOVEMENT_ABILITY            = True
MOVEMENT_ABILITY_KEY               = 'r'
MOVEMENT_ABILITY_COOLDOWN_SECONDS  = 3.0

EXPLORE_NAV_STALL_FRAMES    = 12
EXPLORE_NAV_STALL_MOTION_PX = 0.30

# ── Combat style toggle ───────────────────────────────────────────────────────
# MELEE_MODE = True  → bot always advances into enemies, no kiting or backpedal
# MELEE_MODE = False → ranged behaviour (current default): kite, orbit, flee
MELEE_MODE                  = False
MELEE_FLEE_MINIMAP_DIST     = _px(5)    # px — only retreat if literally overlapping
MELEE_CHASE_MINIMAP_DIST    = _px(20)   # px — A* approach threshold in melee mode
