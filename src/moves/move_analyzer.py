import statistics
from typing import List
import math

import cv2
import mediapipe as mp

from src.utils.geometry import angle_between_points

mp_pose = mp.solutions.pose


class MoveAnalyzer:
    """
    Stateful component that encapsulates 'martial arts move' logic.

    Responsibilities:
    - Maintain stance calibration state across frames.
    - Provide methods to analyze specific movements and draw overlays on the frame.
    - Track per-frame debug metrics that can be drawn or printed.
    """

    # ---- Config defaults ----
    MIN_VIS = 0.5

    # Stance config
    CALIBRATION_FRAMES = 60
    STANCE_TOLERANCE = 0.15
    STANCE_CALIBRATION_DELAY_SEC = 2.0
    
    # Yoi config (torso-normalized: 0 = chest, 1 = hips)
    YOI_WRIST_TORSO_MIN = 0.60   # rough waist-ish lower band
    YOI_WRIST_TORSO_MAX = 0.85   # just above hips
    YOI_MAX_WRIST_SEPARATION_NORM = 0.6  # max L/R wrist separation as fraction of shoulder width


    # Punch config
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

    # Upper block config (torso-normalized: 0 = chest, 1 = hips, negative = above chest)
    UPPER_BLOCK_WRIST_MIN_TORSO_Y = -1.00   # allow wrist well above chest/head
    UPPER_BLOCK_WRIST_MAX_TORSO_Y =  0.10   # anything lower than this is too low for UB

    UPPER_BLOCK_ELBOW_MIN_ANGLE = 80.0      # a bit bent
    UPPER_BLOCK_ELBOW_MAX_ANGLE = 150.0     # but not fully locked out

    # Over-shoulder punch config
    # torso-y: 0 = chest, 1 = hips, negative = above chest/head
    OVER_SHOULDER_WRIST_MIN_TORSO_Y = -1.20   # allow high, over-head
    OVER_SHOULDER_WRIST_MAX_TORSO_Y = -0.10   # still clearly above chest

    OVER_SHOULDER_MIN_SIDE_OFFSET = 0.08      # wrist must be clearly to that side of body center
    OVER_SHOULDER_MIN_ELBOW_ANGLE = 130.0     # fairly extended, not super bent

    def __init__(self, fps: float):
        if fps is None or fps <= 0:
            fps = 30.0
        self.fps = fps
        self.calibration_start_frame = int(self.fps * self.STANCE_CALIBRATION_DELAY_SEC)

        # stance calibration state
        self._calib_feet_widths: List[float] = []
        self._calib_shoulder_widths: List[float] = []
        self._baseline_feet_width: float | None = None
        self._baseline_shoulder_width: float | None = None
        self._baseline_ratio: float | None = None

        # arm length calibration (per side)
        self._right_arm_baseline_len: float | None = None
        self._left_arm_baseline_len: float | None = None

        # per-frame debug data
        self._debug_data: dict[str, float] = {}

        print(
            f"[MoveAnalyzer] Initialized with fps={self.fps:.1f}, "
            f"stance calibration starts at frame {self.calibration_start_frame}"
        )

    # ---------- debug helpers ----------

    def reset_debug_data(self) -> None:
        """Call this once per new frame from the main loop."""
        self._debug_data = {}

    def _record_debug(self, key: str, value) -> None:
        """Store debug metric in a dictionary for later printing."""
        self._debug_data[key] = value

    def get_debug_snapshot(self) -> dict:
        """Return a shallow copy of current debug metrics."""
        return dict(self._debug_data)

    @staticmethod
    def _draw_debug_line(frame_bgr, text: str, x: int, y: int, color):
        cv2.putText(
            frame_bgr,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def _debug_line(
        self,
        frame_bgr,
        label: str,
        value,
        x: int,
        y: int,
        color,
        fmt: str = "{:.2f}",
        store_key: str | None = None,
    ) -> int:
        """
        Convenience helper: record to debug data + draw a line, then return new y for stacking.
        """
        key = store_key or label
        self._record_debug(key, value)

        if isinstance(value, (int, float)):
            text_value = fmt.format(value)
        else:
            text_value = str(value)

        text = f"{label}: {text_value}"
        self._draw_debug_line(frame_bgr, text, x, y, color)
        return y + 20

    # ---------- geometry helpers ----------

    @staticmethod
    def _get_landmark_xy(landmarks, idx, width, height):
        lm = landmarks[idx]
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        return x_px, y_px, lm.visibility

    @staticmethod
    def _horizontal_distance(a_x, b_x) -> float:
        return abs(a_x - b_x)

    @staticmethod
    def _segment_length(a_x, a_y, b_x, b_y) -> float:
        return math.hypot(a_x - b_x, a_y - b_y)

    @staticmethod
    def _body_center_x(landmarks) -> float:
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        return (l_shoulder.x + r_shoulder.x) / 2.0

    # ---------- public analysis methods ----------
    
    def analyze_yoi(self, frame_bgr, landmarks, width, height):
        """
        Detect 'Yoi' position:
          - Both fists together near waist height, roughly centered.
        Uses:
          - torso-normalized wrist height
          - horizontal distance between wrists normalized by shoulder width
        """

        MIN_VIS = self.MIN_VIS

        # ---- torso landmarks ----
        l_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        r_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        l_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
        r_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value

        l_sh = landmarks[l_shoulder_idx]
        r_sh = landmarks[r_shoulder_idx]
        l_hip = landmarks[l_hip_idx]
        r_hip = landmarks[r_hip_idx]

        if (
            l_sh.visibility < MIN_VIS
            or r_sh.visibility < MIN_VIS
            or l_hip.visibility < MIN_VIS
            or r_hip.visibility < MIN_VIS
        ):
            return

        chest_y = (l_sh.y + r_sh.y) / 2.0
        hip_y = (l_hip.y + r_hip.y) / 2.0
        torso_height = hip_y - chest_y
        if torso_height <= 0:
            return

        # Shoulder width in normalized space (for L/R wrist separation scale)
        shoulder_width_norm = abs(r_sh.x - l_sh.x)
        if shoulder_width_norm <= 1e-6:
            return

        # ---- wrists ----
        l_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
        r_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_wrist = landmarks[l_wrist_idx]
        r_wrist = landmarks[r_wrist_idx]

        if l_wrist.visibility < MIN_VIS or r_wrist.visibility < MIN_VIS:
            return

        # Torso-normalized Y for both wrists (0 = chest, 1 = hips)
        l_wrist_torso_y = (l_wrist.y - chest_y) / torso_height
        r_wrist_torso_y = (r_wrist.y - chest_y) / torso_height

        # Horizontal separation between wrists, normalized by shoulder width
        wrist_sep_norm = abs(l_wrist.x - r_wrist.x) / shoulder_width_norm

        # ---- Debug overlays / metrics ----
        y_base = 300
        y_base = self._debug_line(
            frame_bgr,
            "Yoi L wrist torso-y",
            l_wrist_torso_y,
            20,
            y_base,
            (150, 255, 255),
        )
        y_base = self._debug_line(
            frame_bgr,
            "Yoi R wrist torso-y",
            r_wrist_torso_y,
            20,
            y_base,
            (150, 255, 200),
        )
        y_base = self._debug_line(
            frame_bgr,
            "Yoi wrist_sep_norm",
            wrist_sep_norm,
            20,
            y_base,
            (200, 200, 255),
        )

        # ---- Yoi classification ----
        # 1) both wrists in waist-height band
        both_in_waist_band = (
            self.YOI_WRIST_TORSO_MIN <= l_wrist_torso_y <= self.YOI_WRIST_TORSO_MAX
            and self.YOI_WRIST_TORSO_MIN <= r_wrist_torso_y <= self.YOI_WRIST_TORSO_MAX
        )

        # 2) wrists close together horizontally (hands "together")
        wrists_close = wrist_sep_norm <= self.YOI_MAX_WRIST_SEPARATION_NORM

        is_yoi = both_in_waist_band and wrists_close

        if is_yoi:
            label = "Yoi (ready)"
            self._record_debug("yoi_label", label)
            cv2.putText(
                frame_bgr,
                label,
                (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 150, 0),
                2,
                cv2.LINE_AA,
            )



    def analyze_center_punch(self, frame_bgr, landmarks, width, height):
        """
        Detect 'center punch' vs 'chamber at hip' using:
          - elbow angles (for debugging/extra confidence)
          - wrist height relative to torso (shoulders -> hips)
        """
        MIN_VIS = self.MIN_VIS

        # ---- key torso landmarks (normalized) ----
        l_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        r_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        l_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
        r_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value

        l_shoulder = landmarks[l_shoulder_idx]
        r_shoulder = landmarks[r_shoulder_idx]
        l_hip = landmarks[l_hip_idx]
        r_hip = landmarks[r_hip_idx]

        if (
            l_shoulder.visibility < MIN_VIS
            or r_shoulder.visibility < MIN_VIS
            or l_hip.visibility < MIN_VIS
            or r_hip.visibility < MIN_VIS
        ):
            return

        chest_y = (l_shoulder.y + r_shoulder.y) / 2.0
        hip_y = (l_hip.y + r_hip.y) / 2.0
        torso_height = hip_y - chest_y
        if torso_height <= 0:
            return

        # ---- arm indices ----
        r_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        r_elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        r_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        l_elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
        l_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value

        # ========= RIGHT ARM =========
        rsx, rsy, rs_vis = self._get_landmark_xy(landmarks, r_shoulder_idx, width, height)
        rex, rey, re_vis = self._get_landmark_xy(landmarks, r_elbow_idx, width, height)
        rwx, rwy, rw_vis = self._get_landmark_xy(landmarks, r_wrist_idx, width, height)

        right_elbow_angle = None
        if rs_vis > MIN_VIS and re_vis > MIN_VIS and rw_vis > MIN_VIS:
            right_elbow_angle = angle_between_points(
                (rsx, rsy),
                (rex, rey),
                (rwx, rwy),
            )
            if right_elbow_angle is not None:
                # elbow angle debug
                self._record_debug("R elbow angle", right_elbow_angle)
                cv2.circle(frame_bgr, (rex, rey), 5, (0, 255, 0), -1)
                self._draw_debug_line(
                    frame_bgr,
                    f"R elbow: {right_elbow_angle:.1f} deg",
                    rsx + 10,
                    max(rey - 10, 20),
                    (0, 255, 0),
                )

        # Wrist height for right arm (normalized torso coords)
        r_wrist_norm_y = None
        r_wrist = landmarks[r_wrist_idx]
        if r_wrist.visibility > MIN_VIS:
            r_wrist_norm_y = (r_wrist.y - chest_y) / torso_height

        # ========= LEFT ARM =========
        lsx, lsy, ls_vis = self._get_landmark_xy(landmarks, l_shoulder_idx, width, height)
        lex, ley, le_vis = self._get_landmark_xy(landmarks, l_elbow_idx, width, height)
        lwx, lwy, lw_vis = self._get_landmark_xy(landmarks, l_wrist_idx, width, height)

        left_elbow_angle = None
        if ls_vis > MIN_VIS and le_vis > MIN_VIS and lw_vis > MIN_VIS:
            left_elbow_angle = angle_between_points(
                (lsx, lsy),
                (lex, ley),
                (lwx, lwy),
            )
            if left_elbow_angle is not None:
                self._record_debug("L elbow angle", left_elbow_angle)
                cv2.circle(frame_bgr, (lex, ley), 5, (0, 200, 255), -1)
                self._draw_debug_line(
                    frame_bgr,
                    f"L elbow: {left_elbow_angle:.1f} deg",
                    lex + 10,
                    max(ley - 10, 20),
                    (0, 200, 255),
                )

        l_wrist_norm_y = None
        l_wrist = landmarks[l_wrist_idx]
        if l_wrist.visibility > MIN_VIS:
            l_wrist_norm_y = (l_wrist.y - chest_y) / torso_height

        # ---- Overlay torso-relative wrist heights for debugging ----
        y_base = 130
        if r_wrist_norm_y is not None:
            y_base = self._debug_line(
                frame_bgr,
                "R wrist torso-y",
                r_wrist_norm_y,
                20,
                y_base,
                (180, 255, 180),
            )

        if l_wrist_norm_y is not None:
            y_base = self._debug_line(
                frame_bgr,
                "L wrist torso-y",
                l_wrist_norm_y,
                20,
                y_base,
                (180, 255, 255),
            )

        # ---- classify punch vs chamber using torso bands + angle + separation ----
        punch_status_text = None

        def _is_punch_level(torso_y_norm: float | None) -> bool:
            if torso_y_norm is None:
                return False
            return self.PUNCH_TORSO_MIN <= torso_y_norm <= self.PUNCH_TORSO_MAX

        def _is_chamber_level(torso_y_norm: float | None) -> bool:
            if torso_y_norm is None:
                return False
            return self.CHAMBER_TORSO_MIN <= torso_y_norm <= self.CHAMBER_TORSO_MAX

        right_punch_level = _is_punch_level(r_wrist_norm_y)
        right_chamber_level = _is_chamber_level(r_wrist_norm_y)

        left_punch_level = _is_punch_level(l_wrist_norm_y)
        left_chamber_level = _is_chamber_level(l_wrist_norm_y)

        right_angle_extended = (
            right_elbow_angle is not None
            and right_elbow_angle >= self.PUNCH_EXTENDED_MIN_ANGLE
        )
        left_angle_extended = (
            left_elbow_angle is not None
            and left_elbow_angle >= self.PUNCH_EXTENDED_MIN_ANGLE
        )

        MIN_WRIST_SEPARATION = 0.30  # tuned by you

        # Right punch: right wrist mid-torso, left wrist clearly lower (toward hip)
        if (
            right_punch_level
            and left_chamber_level
            and right_angle_extended
            and l_wrist_norm_y is not None
            and r_wrist_norm_y is not None
            and (l_wrist_norm_y - r_wrist_norm_y) >= MIN_WRIST_SEPARATION
        ):
            punch_status_text = "Right center punch: extended"

        # Left punch: left wrist mid-torso, right wrist clearly lower (toward hip)
        elif (
            left_punch_level
            and right_chamber_level
            and left_angle_extended
            and l_wrist_norm_y is not None
            and r_wrist_norm_y is not None
            and (r_wrist_norm_y - l_wrist_norm_y) >= MIN_WRIST_SEPARATION
        ):
            punch_status_text = "Left center punch: extended"

        if punch_status_text:
            self._record_debug("center_punch_label", punch_status_text)
            cv2.putText(
                frame_bgr,
                punch_status_text,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def analyze_upper_block(self, frame_bgr, landmarks, width, height):
        """
        Detect basic left/right upper blocks using:
          - wrist height relative to torso (shoulders -> hips)
          - elbow angle (bent but not fully locked)
        """

        MIN_VIS = self.MIN_VIS

        # ---- torso landmarks for normalization ----
        l_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        r_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        l_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
        r_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value

        l_shoulder = landmarks[l_shoulder_idx]
        r_shoulder = landmarks[r_shoulder_idx]
        l_hip = landmarks[l_hip_idx]
        r_hip = landmarks[r_hip_idx]

        if (
            l_shoulder.visibility < MIN_VIS
            or r_shoulder.visibility < MIN_VIS
            or l_hip.visibility < MIN_VIS
            or r_hip.visibility < MIN_VIS
        ):
            return

        chest_y = (l_shoulder.y + r_shoulder.y) / 2.0
        hip_y = (l_hip.y + r_hip.y) / 2.0
        torso_height = hip_y - chest_y
        if torso_height <= 0:
            return

        # ---- indices per arm ----
        r_elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        r_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
        l_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value

        # ========= RIGHT ARM =========
        r_wrist_norm_y = None
        right_elbow_angle = None

        r_elbow = landmarks[r_elbow_idx]
        r_wrist = landmarks[r_wrist_idx]
        if r_elbow.visibility > MIN_VIS and r_wrist.visibility > MIN_VIS:
            rsx, rsy, rs_vis = self._get_landmark_xy(landmarks, r_shoulder_idx, width, height)
            rex, rey, _ = self._get_landmark_xy(landmarks, r_elbow_idx, width, height)
            rwx, rwy, _ = self._get_landmark_xy(landmarks, r_wrist_idx, width, height)
            if rs_vis > MIN_VIS:
                right_elbow_angle = angle_between_points(
                    (rsx, rsy),
                    (rex, rey),
                    (rwx, rwy),
                )
                self._record_debug("R elbow UB angle", right_elbow_angle)

            r_wrist_norm_y = (r_wrist.y - chest_y) / torso_height

        # ========= LEFT ARM =========
        left_elbow_angle = None
        l_wrist_norm_y = None

        l_elbow = landmarks[l_elbow_idx]
        l_wrist = landmarks[l_wrist_idx]
        if l_elbow.visibility > MIN_VIS and l_wrist.visibility > MIN_VIS:
            lsx, lsy, ls_vis = self._get_landmark_xy(landmarks, l_shoulder_idx, width, height)
            lex, ley, _ = self._get_landmark_xy(landmarks, l_elbow_idx, width, height)
            lwx, lwy, _ = self._get_landmark_xy(landmarks, l_wrist_idx, width, height)
            if ls_vis > MIN_VIS:
                left_elbow_angle = angle_between_points(
                    (lsx, lsy),
                    (lex, ley),
                    (lwx, lwy),
                )
                self._record_debug("L elbow UB angle", left_elbow_angle)

            l_wrist_norm_y = (l_wrist.y - chest_y) / torso_height

        # ---- debug overlays for wrist torso-y ----
        y_base = 170
        if r_wrist_norm_y is not None:
            y_base = self._debug_line(
                frame_bgr,
                "R wrist UB-y",
                r_wrist_norm_y,
                20,
                y_base,
                (255, 200, 180),
            )

        if l_wrist_norm_y is not None:
            y_base = self._debug_line(
                frame_bgr,
                "L wrist UB-y",
                l_wrist_norm_y,
                20,
                y_base,
                (255, 220, 200),
            )

        # ---- helper: does this side look like an upper block? ----
        def _is_upper_block(wrist_norm_y, elbow_angle) -> bool:
            if wrist_norm_y is None or elbow_angle is None:
                return False

            in_height_band = (
                self.UPPER_BLOCK_WRIST_MIN_TORSO_Y
                <= wrist_norm_y
                <= self.UPPER_BLOCK_WRIST_MAX_TORSO_Y
            )
            in_angle_band = (
                self.UPPER_BLOCK_ELBOW_MIN_ANGLE
                <= elbow_angle
                <= self.UPPER_BLOCK_ELBOW_MAX_ANGLE
            )
            return in_height_band and in_angle_band

        right_is_ub = _is_upper_block(r_wrist_norm_y, right_elbow_angle)
        left_is_ub = _is_upper_block(l_wrist_norm_y, left_elbow_angle)

        upper_block_text = None
        if right_is_ub and not left_is_ub:
            upper_block_text = "Right upper block"
        elif left_is_ub and not right_is_ub:
            upper_block_text = "Left upper block"
        elif left_is_ub and right_is_ub:
            upper_block_text = "Upper block (both)"

        if upper_block_text:
            self._record_debug("upper_block_label", upper_block_text)
            cv2.putText(
                frame_bgr,
                upper_block_text,
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def analyze_over_shoulder_punch(self, frame_bgr, landmarks, width, height):
        """
        Detect 'over-the-shoulder' punches and distinguish:
          - same-side (e.g., right arm over right shoulder)
          - cross-body (e.g., right arm over left shoulder)

        Uses:
          - wrist height relative to torso
          - elbow angle (fairly extended)
          - which shoulder the wrist is closest to in X
        """

        MIN_VIS = self.MIN_VIS

        # ---- torso landmarks & normalization ----
        l_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        r_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        l_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
        r_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value

        l_shoulder = landmarks[l_shoulder_idx]
        r_shoulder = landmarks[r_shoulder_idx]
        l_hip = landmarks[l_hip_idx]
        r_hip = landmarks[r_hip_idx]

        if (
            l_shoulder.visibility < MIN_VIS
            or r_shoulder.visibility < MIN_VIS
            or l_hip.visibility < MIN_VIS
            or r_hip.visibility < MIN_VIS
        ):
            return

        chest_y = (l_shoulder.y + r_shoulder.y) / 2.0
        hip_y = (l_hip.y + r_hip.y) / 2.0
        torso_height = hip_y - chest_y
        if torso_height <= 0:
            return

        # We'll use normalized x for shoulder alignment
        l_sh_x = l_shoulder.x
        r_sh_x = r_shoulder.x

        # ---- indices per arm ----
        r_elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        r_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
        l_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value

        # ========= RIGHT ARM =========
        r_elbow = landmarks[r_elbow_idx]
        r_wrist = landmarks[r_wrist_idx]
        r_elbow_angle = None
        r_wrist_norm_y = None
        r_align = None  # "left" or "right"
        r_side_offset = None  # keep for debug

        if r_elbow.visibility > MIN_VIS and r_wrist.visibility > MIN_VIS:
            # pixel coords for elbow/wrist (for angle)
            rsx, rsy, rs_vis = self._get_landmark_xy(landmarks, r_shoulder_idx, width, height)
            rex, rey, _ = self._get_landmark_xy(landmarks, r_elbow_idx, width, height)
            rwx, rwy, _ = self._get_landmark_xy(landmarks, r_wrist_idx, width, height)
            if rs_vis > MIN_VIS:
                r_elbow_angle = angle_between_points(
                    (rsx, rsy),
                    (rex, rey),
                    (rwx, rwy),
                )
                self._record_debug("R elbow OS angle", r_elbow_angle)

            # torso-normalized wrist height
            r_wrist_norm_y = (r_wrist.y - chest_y) / torso_height

            # which shoulder is this wrist closest to in X?
            dist_r_to_left = abs(r_wrist.x - l_sh_x)
            dist_r_to_right = abs(r_wrist.x - r_sh_x)
            r_align = "right" if dist_r_to_right < dist_r_to_left else "left"
            self._record_debug("R OS_align", r_align)

            # still keep side offset vs body center for inspection
            body_center_x = self._body_center_x(landmarks)
            r_side_offset = r_wrist.x - body_center_x

        # ========= LEFT ARM =========
        l_elbow = landmarks[l_elbow_idx]
        l_wrist = landmarks[l_wrist_idx]
        l_elbow_angle = None
        l_wrist_norm_y = None
        l_align = None
        l_side_offset = None

        if l_elbow.visibility > MIN_VIS and l_wrist.visibility > MIN_VIS:
            lsx, lsy, ls_vis = self._get_landmark_xy(landmarks, l_shoulder_idx, width, height)
            lex, ley, _ = self._get_landmark_xy(landmarks, l_elbow_idx, width, height)
            lwx, lwy, _ = self._get_landmark_xy(landmarks, l_wrist_idx, width, height)
            if ls_vis > MIN_VIS:
                l_elbow_angle = angle_between_points(
                    (lsx, lsy),
                    (lex, ley),
                    (lwx, lwy),
                )
                self._record_debug("L elbow OS angle", l_elbow_angle)

            l_wrist_norm_y = (l_wrist.y - chest_y) / torso_height

            dist_l_to_left = abs(l_wrist.x - l_sh_x)
            dist_l_to_right = abs(l_wrist.x - r_sh_x)
            l_align = "left" if dist_l_to_left < dist_l_to_right else "right"
            self._record_debug("L OS_align", l_align)

            body_center_x = self._body_center_x(landmarks)
            l_side_offset = l_wrist.x - body_center_x

        # ---- debug overlays ----
        y_base = 220
        if r_wrist_norm_y is not None and r_side_offset is not None:
            y_base = self._debug_line(
                frame_bgr,
                "R OS-y",
                r_wrist_norm_y,
                20,
                y_base,
                (200, 255, 180),
            )
            y_base = self._debug_line(
                frame_bgr,
                "R OS-dx",
                r_side_offset,
                20,
                y_base,
                (200, 255, 180),
            )

        if l_wrist_norm_y is not None and l_side_offset is not None:
            y_base = self._debug_line(
                frame_bgr,
                "L OS-y",
                l_wrist_norm_y,
                20,
                y_base,
                (220, 255, 200),
            )
            y_base = self._debug_line(
                frame_bgr,
                "L OS-dx",
                l_side_offset,
                20,
                y_base,
                (220, 255, 200),
            )

        # ---- helper: generic over-shoulder condition (height + angle) ----
        def _is_over_shoulder_height_angle(wrist_norm_y, elbow_angle) -> bool:
            if wrist_norm_y is None or elbow_angle is None:
                return False

            in_height_band = (
                self.OVER_SHOULDER_WRIST_MIN_TORSO_Y
                <= wrist_norm_y
                <= self.OVER_SHOULDER_WRIST_MAX_TORSO_Y
            )
            extended = elbow_angle >= self.OVER_SHOULDER_MIN_ELBOW_ANGLE

            return in_height_band and extended

        # Evaluate base OS-ness
        right_os_base = _is_over_shoulder_height_angle(r_wrist_norm_y, r_elbow_angle)
        left_os_base = _is_over_shoulder_height_angle(l_wrist_norm_y, l_elbow_angle)

        os_text = None

        # Right arm OS classification
        if right_os_base and r_align is not None:
            if r_align == "right":
                os_text = "Right over-shoulder (same)"
            else:
                os_text = "Right over-shoulder (cross)"

        # Left arm OS classification (only overwrite if no right label, or choose a display policy)
        if left_os_base and l_align is not None:
            label = "Left over-shoulder (same)" if l_align == "left" else "Left over-shoulder (cross)"
            # if both happen, you can decide which to show; here we append
            if os_text:
                os_text = os_text + " | " + label
            else:
                os_text = label

        if os_text:
            self._record_debug("over_shoulder_label", os_text)
            cv2.putText(
                frame_bgr,
                os_text,
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

    def analyze_horse_stance(self, frame_bgr, landmarks, frame_index: int, width, height):
        """
        Maintain a calibrated horse stance ratio:
          stance_ratio = feet_width / shoulder_width_baseline
        """
        MIN_VIS = self.MIN_VIS
        stance_text_y = 40
        status_text_y = 70

        l_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        r_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        l_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
        r_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value

        ls_x, ls_y, ls_vis = self._get_landmark_xy(landmarks, l_shoulder_idx, width, height)
        rs_x, rs_y, rs_vis = self._get_landmark_xy(landmarks, r_shoulder_idx, width, height)
        la_x, la_y, la_vis = self._get_landmark_xy(landmarks, l_ankle_idx, width, height)
        ra_x, ra_y, ra_vis = self._get_landmark_xy(landmarks, r_ankle_idx, width, height)

        feet_width = None
        shoulder_width = None

        if la_vis > MIN_VIS and ra_vis > MIN_VIS:
            feet_width = self._horizontal_distance(la_x, ra_x)

        if ls_vis > MIN_VIS and rs_vis > MIN_VIS:
            shoulder_width = self._horizontal_distance(ls_x, rs_x)

        # 1) before calibration delay
        if self._baseline_ratio is None and frame_index < self.calibration_start_frame:
            cv2.putText(
                frame_bgr,
                f"Waiting to start stance calibration... (frame {frame_index}/{self.calibration_start_frame})",
                (20, stance_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            return

        # 2) calibration phase
        if (
            self._baseline_ratio is None
            and frame_index >= self.calibration_start_frame
            and feet_width is not None
            and shoulder_width is not None
        ):
            self._calib_feet_widths.append(feet_width)
            self._calib_shoulder_widths.append(shoulder_width)

            cv2.putText(
                frame_bgr,
                f"Calibrating stance... ({len(self._calib_feet_widths)}/{self.CALIBRATION_FRAMES})",
                (20, stance_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if len(self._calib_feet_widths) >= self.CALIBRATION_FRAMES:
                self._baseline_feet_width = statistics.mean(self._calib_feet_widths)
                self._baseline_shoulder_width = statistics.mean(self._calib_shoulder_widths)
                if self._baseline_shoulder_width > 0:
                    self._baseline_ratio = (
                        self._baseline_feet_width / self._baseline_shoulder_width
                    )
                else:
                    self._baseline_ratio = 1.0

                print(
                    f"[MoveAnalyzer] Baseline stance: feet={self._baseline_feet_width:.1f}px, "
                    f"shoulders={self._baseline_shoulder_width:.1f}px, "
                    f"ratio={self._baseline_ratio:.3f}x"
                )
            return

        # 3) after calibration
        if (
            feet_width is not None
            and self._baseline_shoulder_width is not None
            and self._baseline_ratio is not None
        ):
            current_ratio = feet_width / self._baseline_shoulder_width

            cv2.putText(
                frame_bgr,
                f"Stance: {current_ratio:.2f}x shoulders (baseline {self._baseline_ratio:.2f}x)",
                (20, stance_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ratio_vs_baseline = (
                current_ratio / self._baseline_ratio if self._baseline_ratio > 0 else 1.0
            )

            if (1.0 - self.STANCE_TOLERANCE) <= ratio_vs_baseline <= (1.0 + self.STANCE_TOLERANCE):
                status = "Horse stance: OK"
                color = (0, 255, 0)
            elif ratio_vs_baseline < (1.0 - self.STANCE_TOLERANCE):
                status = "Horse stance: too narrow"
                color = (0, 255, 255)
            else:
                status = "Horse stance: too wide"
                color = (0, 165, 255)

            cv2.putText(
                frame_bgr,
                status,
                (20, status_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
