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

# ── Local ONNX models (shipped path — no cloud, CPU-only) ───────────────────
# Weights are produced by training/train_minimap_colab.ipynb and dropped into
# assets/models/. Flags stay OFF until a model beats the heuristic baseline on
# the eval set (acceptance gate in docs/superpowers/specs/2026-07-10-cv-
# recognition-improvements-design.md). Missing weights fail soft either way.
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "models")

LOCAL_MINIMAP_MODEL_ENABLED = True
LOCAL_MINIMAP_MODEL_PATH = os.path.join(_MODELS_DIR, "minimap.onnx")
# Order must match the training notebook's KEEP_CLASSES (= the model's names).
# Trained on 249 real minimap crops the user labeled in Roboflow (incl. an
# `enemy` class the heuristics detect poorly — baseline enemy recall ~0.34).
LOCAL_MINIMAP_MODEL_CLASSES = [
    "enemy", "player", "portal", "dungeon", "enter", "exit",
    "oreGold", "oreGreen", "oreRuby",
]
LOCAL_MINIMAP_MODEL_INPUT_SIZE = 480  # static export size; must match training
LOCAL_MINIMAP_MODEL_CONF = 0.40

LOCAL_SCREEN_MODEL_ENABLED = True
LOCAL_SCREEN_MODEL_PATH = os.path.join(_MODELS_DIR, "screen.onnx")
# On-screen clickable resources (trained on full 1920x1080 loot frames the
# user labeled). Order must match training/train_screen_local.py SCREEN_CLASSES.
# Replaces the noisy find_ore_nodes_on_screen heuristic once the gate passes.
LOCAL_SCREEN_MODEL_CLASSES = ["ore", "item"]
LOCAL_SCREEN_MODEL_INPUT_SIZE = 640   # full frame; ore veins are large
LOCAL_SCREEN_MODEL_CONF = 0.40

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
# How long HP must stay below the flee threshold before actually fleeing. A
# single frame used to be enough, and a single frame is what a spell effect over
# the HP bar produces. Short enough to still run from real danger, long enough
# that one bad read does not flip the state machine.
HP_FLEE_CONFIRM_SECONDS      = 0.6
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

# The minimap prints the player's NAME beside the marker, in the same cream
# tone as the marker itself (measured live: username H22/S102, marker
# H14/S141 -- the item mask takes any saturated bright hue and cannot tell a
# letter from a star). Because the text sits at a fixed offset from the
# player it survived zone resets, so the bot targeted it and stood still
# until the 8s timeout, repeatedly.
#
# Measured over 31 live frames the player UI spans dx -40..+36, dy -17..+29.
# The box is widened rightwards because the sampled name ("Iceberg") is short
# and a longer one runs further. The colour band is checked TOO, so a
# saturated rarity star lying next to the player is still detected -- position
# alone would throw away real loot.
PLAYER_LABEL_DX              = (_px(-52), _px(112))
PLAYER_LABEL_DY              = (_px(-26), _px(38))
PLAYER_LABEL_HUE             = (10, 32)
PLAYER_LABEL_SAT             = (70, 160)
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
# After an on-screen ore is written off as unreachable, stop scanning for a
# moment. Clearing the target alone does not work — the next scan re-finds the
# same node and the bot loops give-up/re-acquire (observed: 10 cycles in 2s on
# one node). Long enough for the bot to move somewhere the node is no longer
# the nearest candidate.
MINING_UNREACHABLE_COOLDOWN_SECONDS   = 6.0
RESOURCE_COLLECT_START_DELAY_SECONDS  = 18.0

LOOT_ITEMS_SCAN_INTERVAL_SECONDS      = 0.10
# After an on-screen item tag is written off as unreachable, stop scanning for
# a moment. Exactly the same fix as MINING_UNREACHABLE_COOLDOWN_SECONDS above,
# for exactly the same reason: clearing the target alone does nothing, because
# the next scan 0.1s later re-finds the same tag. Observed 2026-08-26 -- gave
# up on a tag at (954,533), re-acquired (953,534) one pixel away, and burned
# 9s in two give-up cycles, which then tripped "No attack for 7s -> Vote
# Reset". Long enough for the bot to move somewhere the tag is no longer the
# nearest candidate.
LOOT_ITEMS_UNREACHABLE_COOLDOWN_SECONDS = 6.0
LOOT_ITEMS_CLICK_INTERVAL_SECONDS     = 0.35

# ── minimap-driven resource targeting ────────────────────────────────────
ENABLE_MINIMAP_RESOURCE_TARGETING       = True
MINIMAP_RESOURCE_PICKUP_DIST_PX         = _px(18)   # close enough on minimap to hand off
MINIMAP_RESOURCE_VISIT_MEMORY_SECONDS   = 12.0      # don't revisit failed spots
MINIMAP_RESOURCE_MAX_TRAVEL_SECONDS     = 8.0       # give up after Xs on one target
MINIMAP_RESOURCE_SPOT_MATCH_RADIUS_PX   = _px(8)    # tolerance for "same spot"
MINIMAP_RESOURCE_CLOSE_ATTEMPT_SECONDS  = 4.0       # if close this long, auto-mark attempted
# How long the world may stay static while walking to a resource before the
# target is written off. The approach branch continues past every EXPLORE stuck
# detector, so without this the bot walks into geometry indefinitely — a live
# run held one ore for 7.7s at 0.6px camera shift. Long enough to let a jump
# land and take effect, short enough not to burn a farming cycle.
RT_STATIC_GIVEUP_SECONDS                = 2.0

# How long an ATTACK-state stuck escape (the perpendicular slide) is held before
# the chase may steer again. The stuck block does not continue, so the chase
# would otherwise overwrite a 0.18s escape with a 0.95-1.45s move back into the
# obstruction in the same iteration. Must exceed the chase move duration or the
# escape is pointless; kept just above it so combat still feels responsive.
ATTACK_ESCAPE_COMMIT_SECONDS            = 1.5

# Pause-menu ESC repeat limiter. The auto-ESC path sends ESC, re-checks, and
# sends a second ESC if the menu still looks up — an even number, so a menu that
# survives both just toggles forever. Observed live: ~25s wedged, twice in one
# run, each right after "vote reset unavailable". Backing off is the safe
# failure here; this is the code path behind the Save&Exit incident, so the
# answer to "ESC is not working" must never be "send more ESC".
PAUSE_ESC_MAX_ATTEMPTS                  = 3
PAUSE_ESC_LOOP_WINDOW_SECONDS           = 8.0
PAUSE_ESC_SUPPRESS_SECONDS              = 12.0
# Once standing on/near a minimap-targeted resource, items and ore are both
# collected by pressing F; ore needs a second F this many seconds later to mine.
RESOURCE_PICKUP_SECOND_F_DELAY_SECONDS  = 0.5
# Loot sweep: when combat ends, defer the vote-reset for up to this long while the
# resource targeter is still actively collecting items/ore from the minimap.
LOOT_SWEEP_MAX_SECONDS                  = 12.0

# ── debug snapshots + detection telemetry ────────────────────────────────
ENABLE_DEBUG_SNAPSHOTS              = False
DEBUG_DIR                           = "debug"
# Tuned for multi-hour data runs. At the old 5s periodic interval a 2h session
# produces ~1440 periodic montages on its own, which at DEBUG_MAX_FILES=500
# evicts every EVENT capture (stuck, unreachable, death) — precisely the frames
# worth keeping. Periodics are ambient context; events are evidence. 20s keeps
# ~360 periodics over 2h and leaves the budget for events.
# Montages are ~700KB, so 1500 files is roughly 1GB — check disk before a long
# run, and clear debug/ first.
DEBUG_PERIODIC_INTERVAL_SECONDS     = 20.0
DEBUG_MAX_FILES                     = 1500
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

# ── Targeted loot-frame capture (dev tooling for the on-screen resource model)
# Saves the FULL game frame (no minimap) into a loot/ subfolder whenever the
# bot scans for on-screen ore/items — a dataset dense with clickable resources
# to label an on-screen ore/item detector (the minimap can't be clicked; the
# heuristic on-screen finder is what makes looting miss). Own rate-limit + cap.
CAPTURE_LOOT_FRAMES                 = True
CAPTURE_LOOT_INTERVAL_SECONDS       = 1.5
CAPTURE_LOOT_MAX                    = 300
# Only capture when an ore/item dot sits within this many px of the player on
# the minimap — i.e. the resource is actually near/on-screen and worth
# labeling. Gating on the reliable minimap proximity (not the noisy on-screen
# heuristic) avoids the false-positive frames. ~2x the 18px pickup distance.
CAPTURE_LOOT_NEAR_PX                = _px(40)

ENEMY_DOT_CONFIRM_THRESHOLD         = 2
ENEMY_DOT_MATCH_RADIUS_PX           = _px(6)
ITEM_DOT_CONFIRM_THRESHOLD          = 2
# How soon a vanished resource target counts as "disappeared before we could
# get there" for the resource_target_disappeared_early counter. Was 1.0s,
# which was too tight to see the thing it exists to measure: every drop in the
# 2026-08-27 run landed at 1.0-1.6s, so the counter reported 0 while the
# failure happened ten times in eleven minutes. The counter is how this
# problem gets tracked, so it has to span it.
RESOURCE_TARGET_EARLY_DROP_SECONDS  = 2.5
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
# After a VERIFIED Vote Reset click the zone-select map only auto-opens once
# the vote passes (other players / auto-resolve) — that takes seconds. With the
# default 3.2s wait the sequence declared failure 2s before the map appeared
# and the stray-map net ESC'd it away (2026-07-18 22:05, 10min safe-zone idle).
VOTE_RESET_MAP_WAIT_SECONDS = 10.0
MAP_DETECT_POLL_SECONDS     = 0.18
MAP_INTERACT_RETRY_SECONDS  = 0.80
MAP_OPEN_DELAY_SECONDS      = 0.55
MAP_SELECT_SETTLE_SECONDS   = 0.70
MAP_NODE_CLICK_Y_OFFSET_FRAC = 0.006

# Read the satanic zone name from the HUD instead of hunting a red-glowing
# node by colour. Must be read during GAMEPLAY: the world map is fullscreen
# and hides the HUD entirely (verified -- the reader returns None on every
# captured map frame). False falls back to the colour scan.
MAP_OCR_ENABLED              = True
# Seeking an edge: a cap on iterations, and how long to let the map animate.
# The notch schedule itself is derived from the map width further down.
MAP_NORMALIZE_MAX_STEPS      = 80
MAP_SCROLL_SETTLE_SECONDS    = 0.30
# Scanning the two map edges for the satanic node, used when its name could
# not be read. Measured on a real capture: the active satanic node scores ~615
# while ordinary nodes top out around 305, and the sepia region scores
# NEGATIVE (its nodes are blue, which the scoring penalises). So a winner must
# both clear the ordinary crowd outright and beat the runner-up clearly --
# otherwise a scan that never saw the satanic zone would crown the best
# ordinary node it did see.
SATANIC_SCAN_MIN_SCORE       = 400.0
SATANIC_SCAN_MIN_MARGIN      = 150.0

# A safe zone has no hostile mobs -- that is what makes it safe. So a green
# "Safe Zone" banner read while enemies are still on the minimap is a false
# positive, and acting on one is expensive: release inputs, press F, wait for
# a map that never opens, walk a nudge. Live 2026-08-27 the banner fired while
# the bot was killing ten enemies and kept firing for two minutes (x10
# nudges), stretching that zone to 153.6s against a 35-50s norm.
#
# Safe after a real reset because _do_zone_reset clears last_any_enemy_seen_t,
# so the legitimate case -- teleported onto the waypoint in the new zone's
# protected entrance -- reads as "no enemies for ages" and still fires.
SAFE_ZONE_ENEMY_QUIET_SECONDS = 5.0

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



# ── World-map zone coordinates (recalibrated 2026-08-24) ─────────────────────
# The map is ONE horizontally-scrolling image 3918px wide at 1920x1080; it pans
# 30px per wheel notch and clamps after 1997.7px. Coordinates below are in
# WORLD space -- x on a map panned fully left -- because most zones are off
# screen at any given scroll position. Screen position is `world_x - offset`,
# and the bot only ever parks the map at one of two offsets: 0 (left edge,
# where 40 of these zones are visible) and 1997.7 (right edge, which shows the
# six abyss zones and nothing else). map_scroll.zone_edge_positions() turns
# this table into that split; don't count notches by hand.
#
# Derived by tools/calibrate_map.py + OCR of the map's own labels, not by hand:
# every zone's node was located in each frame it appeared in and the readings
# agreed exactly (zero spread). Two zones had no blue node to detect at capture
# time -- Forest of the Slain was the active satanic zone (red glow) and
# Flooded Plains renders black -- so their nodes come from the label offset,
# which is +32.5px in y with a standard deviation of 0.81px across 44 zones.
#
# This SUPERSEDES _SATANIC_FIXED_NODES_REF above, which the season invalidated:
# only 1 of its 35 entries still lands within 15px of a real node.
WORLD_MAP_WIDTH_PX      = _px(3918)
WORLD_MAP_MAX_SCROLL_PX = _px(1997.7)
WORLD_MAP_PX_PER_NOTCH  = _px(30)

# Panning to an edge, in wheel notches: probe, traverse, probe, traverse...
# Arriving is only ever learned one step late, from a scroll that fails to
# move the map, so the walk alternates. The 3-notch probe buys that proof for
# 90ms -- and a map already parked where it belongs, which is the common case,
# is confirmed on the very first one without touching the traverse step. The
# traverse crosses the whole map at once; overshooting an edge simply clamps,
# and since every step is verified against the image, a step that is too big
# costs nothing and a dropped wheel event just adds an iteration.
MAP_NORMALIZE_STEP_NOTCHES = (
    3,
    int(WORLD_MAP_MAX_SCROLL_PX / WORLD_MAP_PX_PER_NOTCH) + 3,
)

_ZONE_NODES_WORLD_REF = {
    # ── Act 1 ──────────────────────────────────────────────────
    "Outskirts of Inoya": (339, 725),
    "Fields of Battle": (318, 654),
    "The Pumpkin Patch": (221, 594),
    "Woodhill Plains": (305, 548),
    "King's Garden": (387, 488),
    # ── Act 2 ──────────────────────────────────────────────────
    "Crystal Village": (309, 404),
    "Chilling Lake": (104, 411),
    "Arctic Tundra": (197, 369),
    "Snowy Mountains": (282, 306),
    "The Glacial Trail": (446, 371),
    # ── Act 3 ──────────────────────────────────────────────────
    "Corrupted Oasis": (729, 351),
    "Dry Hills": (705, 432),
    "Mos'Arthim Desert": (621, 542),
    "Pyramid Level 1": (759, 608),
    "Pyramid Level 2": (797, 539),
    # ── Act 4 ──────────────────────────────────────────────────
    "Old Mining Village": (240, 896),
    "The Highland Mines": (410, 810),
    "Corrupted Cave": (513, 887),
    "The Nightmare": (525, 783),
    "The Devil's Breach": (630, 747),
    # ── Act 5 ──────────────────────────────────────────────────
    "Mt. Fuji": (770, 842),
    "Misty Swamp": (917, 870),
    "Fuji Coast": (1013, 758),
    "Sea of Karponia": (1110, 834),
    "Temple of Zamjo": (1259, 891),
    # ── Act 6 ──────────────────────────────────────────────────
    "Highland Graveyard": (938, 504),
    "The Cathedral": (1025, 437),
    "Prison Dungeon": (999, 326),
    "Steam Train": (1110, 348),
    "The Depths of Hell": (1134, 515),
    # ── Act 7 ──────────────────────────────────────────────────
    "Deep Space": (1253, 353),
    "Event Horizon": (1310, 461),
    "The Black Hole": (1167, 600),
    "Parallel Dimension": (1221, 649),
    "Subconscious Mind": (1296, 744),
    # ── Act 8 ──────────────────────────────────────────────────
    "Forest of the Slain": (1514, 330),
    "Flooded Plains": (1617, 377),
    "Forgotten Caves": (1764, 459),
    "Camp of Souls": (1623, 567),
    "Hellheim": (1736, 615),
    # ── Abyss, added by the 2026-08 season ──────────────────────────
    "Volcanic Island": (2660, 189),
    "Boreal Island": (2363, 486),
    "Tormented Reef": (2499, 840),
    "Abyss Jungle": (3125, 594),
    "Shipwreck Cove": (3233, 803),
    "Abyss Realm": (3605, 447),
}
ZONE_NODES_WORLD = {
    name: (_px(x), _px(y)) for name, (x, y) in _ZONE_NODES_WORLD_REF.items()
}

# The old name, kept because gui.py builds its zone dropdown from it and
# combat's colour-scan fallback votes map-scroll offsets against it. It now
# points at the recalibrated table: the 35 hand-typed coordinates it used to
# hold were invalidated by the season (1 of 35 within 15px of a real node),
# which also left the offset voting finding a single agreeing pair.
#
# World coordinates are safe for both uses. The map is normalized to its left
# edge before either runs, so on-screen zones ARE at their world x, and zones
# beyond the first screen are skipped by _score_satanic_fixed_node's bounds
# check rather than mis-scored. That also keeps the colour scan away from the
# sepia region entirely, where its red-vs-orange test cannot work.
SATANIC_FIXED_NODES = ZONE_NODES_WORLD

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
