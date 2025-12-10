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
    """

    # ---- Config defaults ----
    MIN_VIS = 0.5

    # Stance config
    CALIBRATION_FRAMES = 60
    STANCE_TOLERANCE = 0.15
    STANCE_CALIBRATION_DELAY_SEC = 2.0

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

        print(
            f"[MoveAnalyzer] Initialized with fps={self.fps:.1f}, "
            f"stance calibration starts at frame {self.calibration_start_frame}"
        )

    # ---------- helper methods ----------

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

    # ---------- public analysis methods ----------

    def analyze_center_punch(self, frame_bgr, landmarks, width, height):
        """
        Detect 'center punch' vs 'chamber at hip' using:
          - elbow angles (for debugging/extra confidence)
          - wrist height relative to torso (shoulders -> hips)

        Coordinate system note:
          - y increases DOWN the image
          - shoulders_y < hips_y
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
            # Torso not well tracked; bail on punch detection for this frame
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
                cv2.putText(
                    frame_bgr,
                    f"R elbow: {right_elbow_angle:.1f} deg",
                    (rex + 10, rey - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(frame_bgr, (rex, rey), 5, (0, 255, 0), -1)

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
                cv2.putText(
                    frame_bgr,
                    f"L elbow: {left_elbow_angle:.1f} deg",
                    (lex + 10, ley - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(frame_bgr, (lex, ley), 5, (0, 200, 255), -1)

        l_wrist_norm_y = None
        l_wrist = landmarks[l_wrist_idx]
        if l_wrist.visibility > MIN_VIS:
            l_wrist_norm_y = (l_wrist.y - chest_y) / torso_height

        # ---- Overlay torso-relative wrist heights for debugging ----
        y_base = 130
        if r_wrist_norm_y is not None:
            cv2.putText(
                frame_bgr,
                f"R wrist torso-y: {r_wrist_norm_y:.2f}",
                (20, y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180),
                1,
                cv2.LINE_AA,
            )
            y_base += 20

        if l_wrist_norm_y is not None:
            cv2.putText(
                frame_bgr,
                f"L wrist torso-y: {l_wrist_norm_y:.2f}",
                (20, y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # ---- classify punch vs chamber using torso bands + (optional) angle ----
        punch_status_text = None

        def _is_punch_level(torso_y_norm: float | None) -> bool:
            if torso_y_norm is None:
                return False
            return (
                self.PUNCH_TORSO_MIN
                <= torso_y_norm
                <= self.PUNCH_TORSO_MAX
            )

        def _is_chamber_level(torso_y_norm: float | None) -> bool:
            if torso_y_norm is None:
                return False
            return (
                self.CHAMBER_TORSO_MIN
                <= torso_y_norm
                <= self.CHAMBER_TORSO_MAX
            )

        # Flags per arm
        right_punch_level = _is_punch_level(r_wrist_norm_y)
        right_chamber_level = _is_chamber_level(r_wrist_norm_y)

        left_punch_level = _is_punch_level(l_wrist_norm_y)
        left_chamber_level = _is_chamber_level(l_wrist_norm_y)

        # Optional: angle checks
        right_angle_extended = (
            right_elbow_angle is not None
            and right_elbow_angle >= self.PUNCH_EXTENDED_MIN_ANGLE
        )
        left_angle_extended = (
            left_elbow_angle is not None
            and left_elbow_angle >= self.PUNCH_EXTENDED_MIN_ANGLE
        )

        # Require a clear vertical separation between wrists (in torso-normalized coords)
        MIN_WRIST_SEPARATION = 0.30  # tweak if needed

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

    def analyze_horse_stance(self, frame_bgr, landmarks, frame_index: int, width, height):
        """
        Maintain a calibrated horse stance ratio:
          stance_ratio = feet_width / shoulder_width_baseline

        Uses:
        - delay before calibration starts (to skip feet-together opening)
        - running average over CALIBRATION_FRAMES
        - tolerance around baseline to label OK / too narrow / too wide
        """
        MIN_VIS = self.MIN_VIS
        stance_text_y = 40
        status_text_y = 70

        # landmark indices
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

        # 1) before we even start calibration (delay period)
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
                    self._baseline_ratio = self._baseline_feet_width / self._baseline_shoulder_width
                else:
                    self._baseline_ratio = 1.0

                print(
                    f"[MoveAnalyzer] Baseline stance: feet={self._baseline_feet_width:.1f}px, "
                    f"shoulders={self._baseline_shoulder_width:.1f}px, "
                    f"ratio={self._baseline_ratio:.3f}x"
                )
            return

        # 3) after calibration: evaluate stance on each frame
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

        right_wrist_norm_y = None
        right_elbow_angle = None

        if rs_vis > MIN_VIS and re_vis > MIN_VIS and rw_vis > MIN_VIS:
            # torso-normalized wrist height (0 = chest, 1 = hips, negative = above chest)
            r_wrist = landmarks[r_wrist_idx]
            if r_wrist.visibility > MIN_VIS:
                right_wrist_norm_y = (r_wrist.y - chest_y) / torso_height

            right_elbow_angle = angle_between_points(
                (rsx, rsy),
                (rex, rey),
                (rwx, rwy),
            )

        # ========= LEFT ARM =========
        lsx, lsy, ls_vis = self._get_landmark_xy(landmarks, l_shoulder_idx, width, height)
        lex, ley, le_vis = self._get_landmark_xy(landmarks, l_elbow_idx, width, height)
        lwx, lwy, lw_vis = self._get_landmark_xy(landmarks, l_wrist_idx, width, height)

        left_wrist_norm_y = None
        left_elbow_angle = None

        if ls_vis > MIN_VIS and le_vis > MIN_VIS and lw_vis > MIN_VIS:
            l_wrist = landmarks[l_wrist_idx]
            if l_wrist.visibility > MIN_VIS:
                left_wrist_norm_y = (l_wrist.y - chest_y) / torso_height

            left_elbow_angle = angle_between_points(
                (lsx, lsy),
                (lex, ley),
                (lwx, lwy),
            )

        # ---- debug overlays for wrist torso-y ----
        y_base = 170
        if right_wrist_norm_y is not None:
            cv2.putText(
                frame_bgr,
                f"R wrist UB-y: {right_wrist_norm_y:.2f}",
                (20, y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 180),
                1,
                cv2.LINE_AA,
            )
            y_base += 20

        if left_wrist_norm_y is not None:
            cv2.putText(
                frame_bgr,
                f"L wrist UB-y: {left_wrist_norm_y:.2f}",
                (20, y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 220, 200),
                1,
                cv2.LINE_AA,
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

        right_is_ub = _is_upper_block(right_wrist_norm_y, right_elbow_angle)
        left_is_ub = _is_upper_block(left_wrist_norm_y, left_elbow_angle)

        upper_block_text = None
        if right_is_ub and not left_is_ub:
            upper_block_text = "Right upper block"
        elif left_is_ub and not right_is_ub:
            upper_block_text = "Left upper block"
        elif left_is_ub and right_is_ub:
            upper_block_text = "Upper block (both)"

        if upper_block_text:
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
