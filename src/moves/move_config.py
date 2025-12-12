"""
Configuration constants for the MoveAnalyzer.

All movement bands, angles, and tolerances live here so they can be
tuned in one place without cluttering move_analyzer.py.

Conventions:
- Torso-normalized Y:
    0.0  ~= chest/shoulders
    1.0  ~= hips
    < 0  = above chest/head
"""

########################################
#           GENERAL                    #
########################################

MIN_VIS = 0.5  # minimum landmark visibility confidence


########################################
#           STANCE CONFIG              #
########################################

CALIBRATION_FRAMES = 60
STANCE_TOLERANCE = 0.15
STANCE_CALIBRATION_DELAY_SEC = 2.0


########################################
#           YOI CONFIG                 #
########################################
# Torso-normalized: 0 = chest, 1 = hips

YOI_WRIST_TORSO_MIN = 0.60   # rough waist-ish lower band
YOI_WRIST_TORSO_MAX = 0.85   # just above hips
YOI_MAX_WRIST_SEPARATION_NORM = 0.60  # max L/R wrist separation as fraction of shoulder width


########################################
#           PUNCH / CHAMBER            #
########################################

# Angle-based hints (currently secondary)
PUNCH_EXTENDED_MIN_ANGLE = 150.0  # elbow angle when arm is essentially extended
PUNCH_CHAMBER_MAX_ANGLE = 120.0   # elbow angle for clearly bent / chambered arm

# Arm length-based punch heuristics (normalized by per-arm baseline)
ARM_EXTENDED_MIN_RATIO = 0.40  # extended if >= this fraction of baseline
ARM_CHAMBER_MAX_RATIO = 0.75   # chambered if <= this fraction of baseline

# Torso-relative bands (0 = chest/shoulders, 1 = hips)
PUNCH_TORSO_MIN = 0.20   # lower bound of "punch" height band
PUNCH_TORSO_MAX = 0.55   # upper bound of "punch" height band

CHAMBER_TORSO_MIN = 0.55  # lower bound of "chamber at hip" band
CHAMBER_TORSO_MAX = 1.05  # upper bound (a bit below hip to allow noise)

# Horizontal wrist separation (normalized by shoulder width)
# Used to avoid misclassifying close-in double-hand moves as punches.
PUNCH_MIN_WRIST_SEPARATION = 0.30


########################################
#           BLOCKS CONFIG              #
########################################
# Upper, outside, inside, and down blocks.
# All use torso-normalized Y and elbow angles, plus height separation
# between blocking and non-blocking arms.

# ---- Upper block (high block) ----
# Torso-normalized: 0 = chest, 1 = hips, negative = above chest/head.
# Tuned so only real high blocks fire, not transitions.

UPPER_BLOCK_WRIST_MIN_TORSO_Y = -1.20   # allow wrist well above chest/head
UPPER_BLOCK_WRIST_MAX_TORSO_Y = -0.20   # must be clearly ABOVE chest line

UPPER_BLOCK_ELBOW_MIN_ANGLE = 95.0      # bent, not hyper-tight
UPPER_BLOCK_ELBOW_MAX_ANGLE = 145.0     # but not fully locked out

UB_MIN_HEIGHT_SEPARATION = 0.40         # blocking arm must be clearly higher than other arm


# ---- Outside center block ----
# Torso-normalized: 0 = chest/shoulders, 1 = hips.
# Based on samples:
#   - blocking wrist torso-y ~ 0.15
#   - non-blocking wrist torso-y ~ 0.66–0.69

OUTSIDE_WRIST_MIN_TORSO_Y = 0.05    # just below chest
OUTSIDE_WRIST_MAX_TORSO_Y = 0.35    # clearly above hip/waist

OUTSIDE_ELBOW_MIN_ANGLE = 85.0      # bent, not straight
OUTSIDE_ELBOW_MAX_ANGLE = 130.0     # avoid hyper-straight or super tight

OUTSIDE_MIN_HEIGHT_SEPARATION = 0.35  # non-blocking wrist clearly lower (toward hip)


# ---- Inside center block ----
# Torso-normalized wrist samples: blocking wrist ~ [-0.03, 0.06]

INSIDE_WRIST_MIN_TORSO_Y = -0.10   # slightly above chest
INSIDE_WRIST_MAX_TORSO_Y =  0.20   # just below chest

# Very bent elbow on blocking arm
INSIDE_ELBOW_MIN_ANGLE = 15.0      # not fully folded, but close
INSIDE_ELBOW_MAX_ANGLE = 60.0      # clearly tighter than a punch or outside block

INSIDE_MIN_HEIGHT_SEPARATION = 0.35  # non-blocking wrist clearly lower (toward hip)


# ---- Down block ----
# Torso-normalized wrist samples: blocking wrist ~ [0.89, 0.96]

DOWN_WRIST_MIN_TORSO_Y = 0.80      # clearly lower than mid-torso
DOWN_WRIST_MAX_TORSO_Y = 1.10      # safely around/below hip

DOWN_ELBOW_MIN_ANGLE = 150.0       # mostly straight
DOWN_ELBOW_MAX_ANGLE = 185.0       # allow slight noise above 180

DOWN_MIN_HEIGHT_SEPARATION = 0.20  # down-block wrist lower than other by this margin


########################################
#       OVER-SHOULDER PUNCH CONFIG     #
########################################
# Torso-normalized Y: 0 = chest, 1 = hips, negative = above chest/head.

OVER_SHOULDER_WRIST_MIN_TORSO_Y = -1.20   # allow high, over-head
OVER_SHOULDER_WRIST_MAX_TORSO_Y =  0.05   # at or slightly above chest/face

# Horizontal offset: how far to that side of the body center the wrist must be.
OVER_SHOULDER_MIN_SIDE_OFFSET = 0.08
# Elbow angle:
# allow both bent (~45°) AND extended (~160°) styles
OVER_SHOULDER_MIN_ELBOW_ANGLE = 40.0
