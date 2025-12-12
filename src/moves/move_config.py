"""
Configuration constants for the MoveAnalyzer.

All movement bands, angles, and tolerances live here so they can be
tuned in one place without cluttering move_analyzer.py.
"""

MIN_VIS = 0.5

# ---- Stance config ----
CALIBRATION_FRAMES = 60
STANCE_TOLERANCE = 0.15
STANCE_CALIBRATION_DELAY_SEC = 2.0

# ---- Yoi config (torso-normalized: 0 = chest, 1 = hips) ----
YOI_WRIST_TORSO_MIN = 0.60   # rough waist-ish lower band
YOI_WRIST_TORSO_MAX = 0.85   # just above hips
YOI_MAX_WRIST_SEPARATION_NORM = 0.6  # max L/R wrist separation as fraction of shoulder width


# ---- Punch config ----
PUNCH_EXTENDED_MIN_ANGLE = 150.0  # (still usable if you want angle too)
PUNCH_CHAMBER_MAX_ANGLE = 120.0

# Arm length-based punch heuristics (normalized ratios)
ARM_EXTENDED_MIN_RATIO = 0.40  # >= 80% of calibrated max length = extended
ARM_CHAMBER_MAX_RATIO = 0.75   # <= 60% of calibrated max length = chambered

# Torso-relative bands (0 = chest/shoulders, 1 = hips)
PUNCH_TORSO_MIN = 0.20   # lower bound of "punch" height band
PUNCH_TORSO_MAX = 0.55   # upper bound of "punch" height band

CHAMBER_TORSO_MIN = 0.55  # lower bound of "chamber at hip" band
CHAMBER_TORSO_MAX = 1.05  # upper bound (a bit below hip to allow noise)

# ---- Upper block config (torso-normalized: 0 = chest, 1 = hips, negative = above chest) ----
# tuned so only real high blocks fire, not transitions
UPPER_BLOCK_WRIST_MIN_TORSO_Y = -1.20   # allow wrist well above chest/head
UPPER_BLOCK_WRIST_MAX_TORSO_Y =  -0.20   # anything lower than this is too low for UB

UPPER_BLOCK_ELBOW_MIN_ANGLE = 95.0      # a bit bent
UPPER_BLOCK_ELBOW_MAX_ANGLE = 145.0     # but not fully locked out
UB_MIN_HEIGHT_SEPARATION = 0.4

# Outside center block bands
OUTSIDE_WRIST_MIN_TORSO_Y = 0.05   # a bit below chest
OUTSIDE_WRIST_MAX_TORSO_Y = 0.35   # above your punch band lower bound
OUTSIDE_ELBOW_MIN_ANGLE   = 85.0
OUTSIDE_ELBOW_MAX_ANGLE   = 130.0
OUTSIDE_MIN_HEIGHT_SEPARATION = 0.35

# ---- Over-shoulder punch config ----
# torso-y: 0 = chest, 1 = hips, negative = above chest/head
OVER_SHOULDER_WRIST_MIN_TORSO_Y = -1.20   # allow high, over-head
OVER_SHOULDER_WRIST_MAX_TORSO_Y = -0.10   # still clearly above chest

OVER_SHOULDER_MIN_SIDE_OFFSET = 0.08      # wrist must be clearly to that side of body center
OVER_SHOULDER_MIN_ELBOW_ANGLE = 130.0     # fairly extended, not super bent