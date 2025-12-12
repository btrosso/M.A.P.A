import statistics
from typing import List
import math

import cv2
import mediapipe as mp

from src.utils.geometry import angle_between_points
from src.moves.kata_sequence import KataStep
from . import move_config as cfg

mp_pose = mp.solutions.pose


class MoveAnalyzer:
    """
    Stateful component that encapsulates 'martial arts move' logic.

    Responsibilities:
    - Maintain stance calibration state across frames.
    - Provide methods to analyze specific movements and draw overlays on the frame.
    - Track per-frame debug metrics that can be drawn or printed.
    """

    ########################################
    #           GENERAL                    #
    ########################################

    MIN_VIS = cfg.MIN_VIS


    ########################################
    #           STANCE CONFIG              #
    ########################################

    CALIBRATION_FRAMES = cfg.CALIBRATION_FRAMES
    STANCE_TOLERANCE = cfg.STANCE_TOLERANCE
    STANCE_CALIBRATION_DELAY_SEC = cfg.STANCE_CALIBRATION_DELAY_SEC


    ########################################
    #           YOI CONFIG                 #
    ########################################

    YOI_WRIST_TORSO_MIN = cfg.YOI_WRIST_TORSO_MIN
    YOI_WRIST_TORSO_MAX = cfg.YOI_WRIST_TORSO_MAX
    YOI_MAX_WRIST_SEPARATION_NORM = cfg.YOI_MAX_WRIST_SEPARATION_NORM


    ########################################
    #        PUNCH / CHAMBER CONFIG        #
    ########################################

    # Angle hints
    PUNCH_EXTENDED_MIN_ANGLE = cfg.PUNCH_EXTENDED_MIN_ANGLE
    PUNCH_CHAMBER_MAX_ANGLE = cfg.PUNCH_CHAMBER_MAX_ANGLE

    # Normalized arm length ratios
    ARM_EXTENDED_MIN_RATIO = cfg.ARM_EXTENDED_MIN_RATIO
    ARM_CHAMBER_MAX_RATIO = cfg.ARM_CHAMBER_MAX_RATIO

    # Torso-relative bands
    PUNCH_TORSO_MIN = cfg.PUNCH_TORSO_MIN
    PUNCH_TORSO_MAX = cfg.PUNCH_TORSO_MAX

    CHAMBER_TORSO_MIN = cfg.CHAMBER_TORSO_MIN
    CHAMBER_TORSO_MAX = cfg.CHAMBER_TORSO_MAX

    # Horizontal wrist separation
    PUNCH_MIN_WRIST_SEPARATION = cfg.PUNCH_MIN_WRIST_SEPARATION


    ########################################
    #           BLOCKS CONFIG              #
    ########################################

    # Upper block (high block)
    UPPER_BLOCK_WRIST_MIN_TORSO_Y = cfg.UPPER_BLOCK_WRIST_MIN_TORSO_Y
    UPPER_BLOCK_WRIST_MAX_TORSO_Y = cfg.UPPER_BLOCK_WRIST_MAX_TORSO_Y

    UPPER_BLOCK_ELBOW_MIN_ANGLE = cfg.UPPER_BLOCK_ELBOW_MIN_ANGLE
    UPPER_BLOCK_ELBOW_MAX_ANGLE = cfg.UPPER_BLOCK_ELBOW_MAX_ANGLE

    UB_MIN_HEIGHT_SEPARATION = cfg.UB_MIN_HEIGHT_SEPARATION

    # Outside center block
    OUTSIDE_WRIST_MIN_TORSO_Y = cfg.OUTSIDE_WRIST_MIN_TORSO_Y
    OUTSIDE_WRIST_MAX_TORSO_Y = cfg.OUTSIDE_WRIST_MAX_TORSO_Y
    OUTSIDE_ELBOW_MIN_ANGLE = cfg.OUTSIDE_ELBOW_MIN_ANGLE
    OUTSIDE_ELBOW_MAX_ANGLE = cfg.OUTSIDE_ELBOW_MAX_ANGLE
    OUTSIDE_MIN_HEIGHT_SEPARATION = cfg.OUTSIDE_MIN_HEIGHT_SEPARATION

    # Inside center block
    INSIDE_WRIST_MIN_TORSO_Y = cfg.INSIDE_WRIST_MIN_TORSO_Y
    INSIDE_WRIST_MAX_TORSO_Y = cfg.INSIDE_WRIST_MAX_TORSO_Y
    INSIDE_ELBOW_MIN_ANGLE = cfg.INSIDE_ELBOW_MIN_ANGLE
    INSIDE_ELBOW_MAX_ANGLE = cfg.INSIDE_ELBOW_MAX_ANGLE
    INSIDE_MIN_HEIGHT_SEPARATION = cfg.INSIDE_MIN_HEIGHT_SEPARATION

    # Down block
    DOWN_WRIST_MIN_TORSO_Y = cfg.DOWN_WRIST_MIN_TORSO_Y
    DOWN_WRIST_MAX_TORSO_Y = cfg.DOWN_WRIST_MAX_TORSO_Y
    DOWN_ELBOW_MIN_ANGLE = cfg.DOWN_ELBOW_MIN_ANGLE
    DOWN_ELBOW_MAX_ANGLE = cfg.DOWN_ELBOW_MAX_ANGLE
    DOWN_MIN_HEIGHT_SEPARATION = cfg.DOWN_MIN_HEIGHT_SEPARATION


    ########################################
    #     OVER-SHOULDER PUNCH CONFIG       #
    ########################################

    OVER_SHOULDER_WRIST_MIN_TORSO_Y = cfg.OVER_SHOULDER_WRIST_MIN_TORSO_Y
    OVER_SHOULDER_WRIST_MAX_TORSO_Y = cfg.OVER_SHOULDER_WRIST_MAX_TORSO_Y

    OVER_SHOULDER_MIN_SIDE_OFFSET = cfg.OVER_SHOULDER_MIN_SIDE_OFFSET
    OVER_SHOULDER_MIN_ELBOW_ANGLE = cfg.OVER_SHOULDER_MIN_ELBOW_ANGLE


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

        # HUD state (per frame)
        self._hud_metrics: list[str] = []
        self._hud_moves: list[str] = []
        # per-frame debug data - debug snapshot store for your "d" key
        self._debug_data: dict[str, float] = {}
        
        print(
            f"[MoveAnalyzer] Initialized with fps={self.fps:.1f}, "
            f"stance calibration starts at frame {self.calibration_start_frame}"
        )
        
        self._kata_steps = [
            KataStep("yoi", self.analyze_yoi, required_sides=set(), confirm_frames=5),
            KataStep("upper_block", self.analyze_upper_block, required_sides={"left","right"}, confirm_frames=3),
            KataStep("outside_block", self.analyze_outside_block, required_sides={"left","right"}, confirm_frames=3),
            KataStep("inside_block", self.analyze_inside_block, required_sides={"left","right"}, confirm_frames=3),
            KataStep("down_block", self.analyze_down_block, required_sides={"left","right"}, confirm_frames=3),
            KataStep("center_punch", self.analyze_center_punch, required_sides={"left","right"}, confirm_frames=2),
            # add trap, over-shoulder, etc later
        ]
        self._kata_index = 0
        self._kata_steps[0].reset_runtime()

    # ---------- HUD helpers ----------

    def reset_hud(self):
        """Call this once per frame before running analyzers."""
        self._hud_metrics = []
        self._hud_moves = []

    def _add_metric_line(self, label: str, value):
        """Record a metric line to be drawn later in the HUD."""
        if isinstance(value, float):
            text = f"{label}: {value:.3f}"
        else:
            text = f"{label}: {value}"
        self._hud_metrics.append(text)

    def _add_move_label(self, label: str):
        """Record a movement label to be drawn below the metrics."""
        if label not in self._hud_moves:
            self._hud_moves.append(label)

    def draw_hud(self, frame_bgr):
        """
        Render metrics (green) and movement labels (blue)
        in the top-right corner in a stacked HUD.
        """
        h, w, _ = frame_bgr.shape

        # We'll anchor text ~20px from the right edge.
        margin_x = 20
        x_right = w - margin_x

        # Start a bit down from the top
        y = 30

        # Metrics (green)
        for line in self._hud_metrics:
            # Get text size so we can right-align
            (text_w, text_h), _ = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )
            x = x_right - text_w

            cv2.putText(
                frame_bgr,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),   # green
                1,
                cv2.LINE_AA,
            )
            y += 18  # vertical spacing

        # Small gap between metrics and move labels
        y += 10

        # Movement labels (blue, slightly bigger)
        for label in self._hud_moves:
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )
            x = x_right - text_w

            cv2.putText(
                frame_bgr,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),   # blue
                2,
                cv2.LINE_AA,
            )
            y += 24


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

    def _debug_line(self, frame_bgr, label: str, value):
        """
        Legacy helper used throughout analyzers.

        Now:
          - does NOT draw directly on the frame
          - instead, records a metric line for the HUD

        Returns the same y value so existing call sites don't care.
        """
        self._add_metric_line(label, value)


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
        return None

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
        self._debug_line(frame_bgr, "Yoi L wrist torso-y", l_wrist_torso_y,)
        self._debug_line(frame_bgr, "Yoi R wrist torso-y", r_wrist_torso_y,)
        self._debug_line(frame_bgr, "Yoi wrist_sep_norm", wrist_sep_norm,)

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
            self._add_move_label(label)
            return ("yoi", None)
        
        return None
    
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
        if r_wrist_norm_y is not None:
            self._debug_line(frame_bgr, "R wrist UB-y", r_wrist_norm_y,)

        if l_wrist_norm_y is not None:
            self._debug_line(frame_bgr, "L wrist UB-y", l_wrist_norm_y,)

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
        side_text = None
        
        # require a clear vertical separation between wrists
        if r_wrist_norm_y is not None and l_wrist_norm_y is not None:
            # positive: right wrist is lower on the body than left
            diff_right_minus_left = r_wrist_norm_y - l_wrist_norm_y
            diff_left_minus_right = l_wrist_norm_y - r_wrist_norm_y

            # Right upper block: right wrist high (UB), clearly above left wrist
            if (
                right_is_ub
                and not left_is_ub
                and diff_left_minus_right >= self.UB_MIN_HEIGHT_SEPARATION
            ):
                upper_block_text = "Right upper block"
                side_text = "right"

            # Left upper block: left wrist high (UB), clearly above right wrist
            elif (
                left_is_ub
                and not right_is_ub
                and diff_right_minus_left >= self.UB_MIN_HEIGHT_SEPARATION
            ):
                upper_block_text = "Left upper block"
                side_text = "left"

            # (Optional) both arms up overhead in some special posture
            elif left_is_ub and right_is_ub:
                upper_block_text = "Upper block (both)"
                side_text = "both"

        if upper_block_text:
            self._record_debug("upper_block_label", upper_block_text)
            self._add_move_label(upper_block_text)
            return ("upper_block", side_text)
        
        return None

    def analyze_outside_block(self, frame_bgr, landmarks, width, height):
        """
        Detect outside center blocks:

          - Left outside block:
              * left wrist at mid torso (~0.1-0.3 torso-y)
              * right wrist near hip (chamber band)
              * left elbow bent in a reasonable range

          - Right outside block:
              * right wrist at mid torso
              * left wrist near hip
              * right elbow bent

        Uses:
          - torso-normalized wrist height
          - elbow angle
          - wrist-to-wrist vertical separation
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

        # ---- elbow & wrist indices ----
        l_elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
        r_elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        l_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
        r_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_elbow = landmarks[l_elbow_idx]
        r_elbow = landmarks[r_elbow_idx]
        l_wrist = landmarks[l_wrist_idx]
        r_wrist = landmarks[r_wrist_idx]

        if (
            l_elbow.visibility < MIN_VIS
            or r_elbow.visibility < MIN_VIS
            or l_wrist.visibility < MIN_VIS
            or r_wrist.visibility < MIN_VIS
        ):
            return

        # ---- torso-normalized wrist heights ----
        l_wrist_torso_y = (l_wrist.y - chest_y) / torso_height
        r_wrist_torso_y = (r_wrist.y - chest_y) / torso_height

        # ---- elbow angles (in pixels for stability) ----
        lsx, lsy, ls_vis = self._get_landmark_xy(landmarks, l_shoulder_idx, width, height)
        rsx, rsy, rs_vis = self._get_landmark_xy(landmarks, r_shoulder_idx, width, height)
        lex, ley, _ = self._get_landmark_xy(landmarks, l_elbow_idx, width, height)
        rex, rey, _ = self._get_landmark_xy(landmarks, r_elbow_idx, width, height)
        lwx, lwy, _ = self._get_landmark_xy(landmarks, l_wrist_idx, width, height)
        rwx, rwy, _ = self._get_landmark_xy(landmarks, r_wrist_idx, width, height)

        left_elbow_angle = None
        right_elbow_angle = None

        if ls_vis > MIN_VIS:
            left_elbow_angle = angle_between_points(
                (lsx, lsy),
                (lex, ley),
                (lwx, lwy),
            )
            self._record_debug("L elbow OUT angle", left_elbow_angle)

        if rs_vis > MIN_VIS:
            right_elbow_angle = angle_between_points(
                (rsx, rsy),
                (rex, rey),
                (rwx, rwy),
            )
            self._record_debug("R elbow OUT angle", right_elbow_angle)

        # ---- Debug overlays ----
        self._debug_line(            frame_bgr,
            "OUT L wrist torso-y",
            l_wrist_torso_y,
        )
        self._debug_line(            frame_bgr,
            "OUT R wrist torso-y",
            r_wrist_torso_y,
        )

        # ---- helper bands ----
        def _is_out_block_band(torso_y, elbow_angle) -> bool:
            if torso_y is None or elbow_angle is None:
                return False
            in_height = (
                self.OUTSIDE_WRIST_MIN_TORSO_Y
                <= torso_y
                <= self.OUTSIDE_WRIST_MAX_TORSO_Y
            )
            in_angle = (
                self.OUTSIDE_ELBOW_MIN_ANGLE
                <= elbow_angle
                <= self.OUTSIDE_ELBOW_MAX_ANGLE
            )
            return in_height and in_angle

        def _is_chamber_band(torso_y) -> bool:
            if torso_y is None:
                return False
            return self.CHAMBER_TORSO_MIN <= torso_y <= self.CHAMBER_TORSO_MAX

        left_is_out = _is_out_block_band(l_wrist_torso_y, left_elbow_angle)
        right_is_out = _is_out_block_band(r_wrist_torso_y, right_elbow_angle)

        # How much lower is one wrist vs the other?
        # (bigger torso-y = lower on body)
        diff_right_minus_left = r_wrist_torso_y - l_wrist_torso_y
        diff_left_minus_right = l_wrist_torso_y - r_wrist_torso_y

        label = None
        side_text = None

        # Left outside block:
        # - left wrist in outside band
        # - right wrist in chamber band (hip-ish)
        # - right wrist clearly lower than left
        if (
            left_is_out
            and not right_is_out
            and _is_chamber_band(r_wrist_torso_y)
            and diff_right_minus_left >= self.OUTSIDE_MIN_HEIGHT_SEPARATION
        ):
            label = "Left outside block"
            side_text = "left"

        # Right outside block:
        # - right wrist in outside band
        # - left wrist in chamber band
        # - left wrist clearly lower than right
        elif (
            right_is_out
            and not left_is_out
            and _is_chamber_band(l_wrist_torso_y)
            and diff_left_minus_right >= self.OUTSIDE_MIN_HEIGHT_SEPARATION
        ):
            label = "Right outside block"
            side_text = "right"

        if label:
            self._record_debug("outside_block_label", label)
            self._add_move_label(label)
            return ("outside_block", side_text)
        
        return None

    def analyze_inside_block(self, frame_bgr, landmarks, width, height):
        """
        Detect inside center blocks:

          - Left inside block:
              * left wrist near chest line (slightly above/below)
              * right wrist near hip (chamber)
              * left elbow very bent (tight inward block)

          - Right inside block:
              * right wrist near chest line
              * left wrist near hip
              * right elbow very bent
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

        # ---- indices ----
        l_elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
        r_elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        l_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
        r_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_elbow = landmarks[l_elbow_idx]
        r_elbow = landmarks[r_elbow_idx]
        l_wrist = landmarks[l_wrist_idx]
        r_wrist = landmarks[r_wrist_idx]

        if (
            l_elbow.visibility < MIN_VIS
            or r_elbow.visibility < MIN_VIS
            or l_wrist.visibility < MIN_VIS
            or r_wrist.visibility < MIN_VIS
        ):
            return

        # ---- torso-normalized wrist heights ----
        l_wrist_torso_y = (l_wrist.y - chest_y) / torso_height
        r_wrist_torso_y = (r_wrist.y - chest_y) / torso_height

        # ---- elbow angles ----
        lsx, lsy, ls_vis = self._get_landmark_xy(landmarks, l_shoulder_idx, width, height)
        rsx, rsy, rs_vis = self._get_landmark_xy(landmarks, r_shoulder_idx, width, height)
        lex, ley, _ = self._get_landmark_xy(landmarks, l_elbow_idx, width, height)
        rex, rey, _ = self._get_landmark_xy(landmarks, r_elbow_idx, width, height)
        lwx, lwy, _ = self._get_landmark_xy(landmarks, l_wrist_idx, width, height)
        rwx, rwy, _ = self._get_landmark_xy(landmarks, r_wrist_idx, width, height)

        left_elbow_angle = None
        right_elbow_angle = None

        if ls_vis > MIN_VIS:
            left_elbow_angle = angle_between_points(
                (lsx, lsy),
                (lex, ley),
                (lwx, lwy),
            )
            self._record_debug("L elbow IN angle", left_elbow_angle)

        if rs_vis > MIN_VIS:
            right_elbow_angle = angle_between_points(
                (rsx, rsy),
                (rex, rey),
                (rwx, rwy),
            )
            self._record_debug("R elbow IN angle", right_elbow_angle)

        # ---- debug overlays ----
        self._debug_line(            frame_bgr,
            "IN L wrist torso-y",
            l_wrist_torso_y,
        )
        self._debug_line(            frame_bgr,
            "IN R wrist torso-y",
            r_wrist_torso_y,
        )

        # ---- helper bands ----
        def _is_inside_band(torso_y, elbow_angle) -> bool:
            if torso_y is None or elbow_angle is None:
                return False
            in_height = (
                self.INSIDE_WRIST_MIN_TORSO_Y
                <= torso_y
                <= self.INSIDE_WRIST_MAX_TORSO_Y
            )
            in_angle = (
                self.INSIDE_ELBOW_MIN_ANGLE
                <= elbow_angle
                <= self.INSIDE_ELBOW_MAX_ANGLE
            )
            return in_height and in_angle

        def _is_chamber_band(torso_y) -> bool:
            if torso_y is None:
                return False
            return self.CHAMBER_TORSO_MIN <= torso_y <= self.CHAMBER_TORSO_MAX

        left_is_inside = _is_inside_band(l_wrist_torso_y, left_elbow_angle)
        right_is_inside = _is_inside_band(r_wrist_torso_y, right_elbow_angle)

        # vertical differences (positive means that side is lower on body)
        diff_right_minus_left = r_wrist_torso_y - l_wrist_torso_y
        diff_left_minus_right = l_wrist_torso_y - r_wrist_torso_y

        label = None
        side_text = None

        # Left inside block: left wrist at chest line, right wrist at hip
        if (
            left_is_inside
            and not right_is_inside
            and _is_chamber_band(r_wrist_torso_y)
            and diff_right_minus_left >= self.INSIDE_MIN_HEIGHT_SEPARATION
        ):
            label = "Left inside block"
            side_text = "left"

        # Right inside block: right wrist at chest line, left wrist at hip
        elif (
            right_is_inside
            and not left_is_inside
            and _is_chamber_band(l_wrist_torso_y)
            and diff_left_minus_right >= self.INSIDE_MIN_HEIGHT_SEPARATION
        ):
            label = "Right inside block"
            side_text = "right"

        if label:
            self._record_debug("inside_block_label", label)
            self._add_move_label(label)
            return ("inside_block", side_text)
        
        return None

    def analyze_down_block(self, frame_bgr, landmarks, width, height):
        """
        Detect down blocks:

          - Left down block:
              * left wrist low near hip/knee (high torso-y ~ 0.8-1.1)
              * right wrist nearer to normal chamber height
              * left elbow mostly straight

          - Right down block:
              * right wrist low
              * left wrist near chamber
              * right elbow mostly straight
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

        # ---- indices ----
        l_elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
        r_elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        l_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
        r_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_elbow = landmarks[l_elbow_idx]
        r_elbow = landmarks[r_elbow_idx]
        l_wrist = landmarks[l_wrist_idx]
        r_wrist = landmarks[r_wrist_idx]

        if (
            l_elbow.visibility < MIN_VIS
            or r_elbow.visibility < MIN_VIS
            or l_wrist.visibility < MIN_VIS
            or r_wrist.visibility < MIN_VIS
        ):
            return

        # ---- torso-normalized wrist heights ----
        l_wrist_torso_y = (l_wrist.y - chest_y) / torso_height
        r_wrist_torso_y = (r_wrist.y - chest_y) / torso_height

        # ---- elbow angles ----
        lsx, lsy, ls_vis = self._get_landmark_xy(landmarks, l_shoulder_idx, width, height)
        rsx, rsy, rs_vis = self._get_landmark_xy(landmarks, r_shoulder_idx, width, height)
        lex, ley, _ = self._get_landmark_xy(landmarks, l_elbow_idx, width, height)
        rex, rey, _ = self._get_landmark_xy(landmarks, r_elbow_idx, width, height)
        lwx, lwy, _ = self._get_landmark_xy(landmarks, l_wrist_idx, width, height)
        rwx, rwy, _ = self._get_landmark_xy(landmarks, r_wrist_idx, width, height)

        left_elbow_angle = None
        right_elbow_angle = None

        if ls_vis > MIN_VIS:
            left_elbow_angle = angle_between_points(
                (lsx, lsy),
                (lex, ley),
                (lwx, lwy),
            )
            self._record_debug("L elbow DOWN angle", left_elbow_angle)

        if rs_vis > MIN_VIS:
            right_elbow_angle = angle_between_points(
                (rsx, rsy),
                (rex, rey),
                (rwx, rwy),
            )
            self._record_debug("R elbow DOWN angle", right_elbow_angle)

        # ---- debug overlays ----
        self._debug_line(            frame_bgr, 
            "DOWN L wrist torso-y", 
            l_wrist_torso_y
        )
        self._debug_line(            frame_bgr,
            "DOWN R wrist torso-y",
            r_wrist_torso_y,
        )

        # ---- helpers ----
        def _is_down_band(torso_y, elbow_angle) -> bool:
            if torso_y is None or elbow_angle is None:
                return False
            in_height = (
                self.DOWN_WRIST_MIN_TORSO_Y
                <= torso_y
                <= self.DOWN_WRIST_MAX_TORSO_Y
            )
            in_angle = (
                self.DOWN_ELBOW_MIN_ANGLE
                <= elbow_angle
                <= self.DOWN_ELBOW_MAX_ANGLE
            )
            return in_height and in_angle

        def _is_chamber_band(torso_y) -> bool:
            if torso_y is None:
                return False
            return self.CHAMBER_TORSO_MIN <= torso_y <= self.CHAMBER_TORSO_MAX

        left_is_down = _is_down_band(l_wrist_torso_y, left_elbow_angle)
        right_is_down = _is_down_band(r_wrist_torso_y, right_elbow_angle)

        diff_right_minus_left = r_wrist_torso_y - l_wrist_torso_y
        diff_left_minus_right = l_wrist_torso_y - r_wrist_torso_y

        label = None
        side_text = None

        # Left down block
        if (
            left_is_down
            and not right_is_down
            and _is_chamber_band(r_wrist_torso_y)
            and diff_right_minus_left <= -self.DOWN_MIN_HEIGHT_SEPARATION
        ):
            # left wrist should be LOWER (bigger torso-y) than right,
            # but our diff is (right - left),
            # so for left lower, diff_right_minus_left is negative.
            label = "Left down block"
            side_text = "left"

        # Right down block
        elif (
            right_is_down
            and not left_is_down
            and _is_chamber_band(l_wrist_torso_y)
            and diff_left_minus_right <= -self.DOWN_MIN_HEIGHT_SEPARATION
        ):
            # similarly, right lower ⇒ (left - right) negative
            label = "Right down block"
            side_text = "right"

        if label:
            self._record_debug("down_block_label", label)
            self._add_move_label(label)
            return ("down_block", side_text)
        
        return None
    
    def analyze_center_punch(self, frame_bgr, landmarks, width, height):
        """
        Detect simple left/right center punches, using:

          - Shoulder->wrist length normalized by a per-arm baseline
          - Torso-normalized wrist height:
              * punch: mid torso band
              * chamber: hip band
          - Wrist horizontal separation to avoid traps / double-hand moves
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

        # shoulder width in normalized coords (for wrist separation scaling)
        shoulder_width_norm = abs(r_sh.x - l_sh.x)
        if shoulder_width_norm <= 1e-6:
            return

        # ---- indices per arm ----
        r_elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        r_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
        l_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value

        # ========= RIGHT ARM =========
        r_elbow = landmarks[r_elbow_idx]
        r_wrist = landmarks[r_wrist_idx]

        right_len = None
        right_len_norm = None
        right_wrist_torso_y = None
        right_elbow_angle = None

        if r_elbow.visibility > MIN_VIS and r_wrist.visibility > MIN_VIS:
            rsx, rsy, rs_vis = self._get_landmark_xy(landmarks, r_shoulder_idx, width, height)
            rex, rey, _ = self._get_landmark_xy(landmarks, r_elbow_idx, width, height)
            rwx, rwy, _ = self._get_landmark_xy(landmarks, r_wrist_idx, width, height)

            if rs_vis > MIN_VIS:
                # elbow angle mostly for debugging
                right_elbow_angle = angle_between_points(
                    (rsx, rsy),
                    (rex, rey),
                    (rwx, rwy),
                )
                self._record_debug("R elbow PUNCH angle", right_elbow_angle)

                # shoulder->wrist length
                right_len = self._segment_length(rsx, rsy, rwx, rwy)
                if right_len is not None:
                    if (
                        self._right_arm_baseline_len is None
                        or right_len > self._right_arm_baseline_len
                    ):
                        self._right_arm_baseline_len = right_len

                    if self._right_arm_baseline_len:
                        right_len_norm = right_len / self._right_arm_baseline_len

            # torso-normalized Y for right wrist
            right_wrist_torso_y = (r_wrist.y - chest_y) / torso_height

        # ========= LEFT ARM =========
        l_elbow = landmarks[l_elbow_idx]
        l_wrist = landmarks[l_wrist_idx]

        left_len = None
        left_len_norm = None
        left_wrist_torso_y = None
        left_elbow_angle = None

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
                self._record_debug("L elbow PUNCH angle", left_elbow_angle)

                left_len = self._segment_length(lsx, lsy, lwx, lwy)
                if left_len is not None:
                    if (
                        self._left_arm_baseline_len is None
                        or left_len > self._left_arm_baseline_len
                    ):
                        self._left_arm_baseline_len = left_len

                    if self._left_arm_baseline_len:
                        left_len_norm = left_len / self._left_arm_baseline_len

            left_wrist_torso_y = (l_wrist.y - chest_y) / torso_height

        # ---- wrist separation (normalized) ----
        wrist_sep_norm = None
        if l_wrist is not None and r_wrist is not None:
            wrist_sep_norm = abs(l_wrist.x - r_wrist.x) / shoulder_width_norm

        # ---- HUD metrics ----
        if right_len_norm is not None:
            self._debug_line(frame_bgr, "R len norm", right_len_norm,)
        if left_len_norm is not None:
            self._debug_line(frame_bgr, "L len norm", left_len_norm,)
        if right_wrist_torso_y is not None:
            self._debug_line(frame_bgr, "R wrist torso-y", right_wrist_torso_y,)
        if left_wrist_torso_y is not None:
            self._debug_line(frame_bgr, "L wrist torso-y", left_wrist_torso_y,)
        if wrist_sep_norm is not None:
            self._debug_line(frame_bgr, "PUNCH wrist_sep_norm", wrist_sep_norm,)

        # ---- helpers ----
        def _is_punch_side(len_norm, wrist_torso_y) -> bool:
            if len_norm is None or wrist_torso_y is None:
                return False
            return (
                len_norm >= self.ARM_EXTENDED_MIN_RATIO
                and self.PUNCH_TORSO_MIN <= wrist_torso_y <= self.PUNCH_TORSO_MAX
            )

        def _is_chamber_side(len_norm, wrist_torso_y) -> bool:
            if len_norm is None or wrist_torso_y is None:
                return False
            return (
                len_norm <= self.ARM_CHAMBER_MAX_RATIO
                and self.CHAMBER_TORSO_MIN <= wrist_torso_y <= self.CHAMBER_TORSO_MAX
            )

        right_is_punch = _is_punch_side(right_len_norm, right_wrist_torso_y)
        left_is_punch = _is_punch_side(left_len_norm, left_wrist_torso_y)

        right_is_chamber = _is_chamber_side(right_len_norm, right_wrist_torso_y)
        left_is_chamber = _is_chamber_side(left_len_norm, left_wrist_torso_y)

        label = None
        side_text = None

        if (
            wrist_sep_norm is not None
            and wrist_sep_norm >= self.PUNCH_MIN_WRIST_SEPARATION
        ):
            # Right punch, left chamber
            if right_is_punch and left_is_chamber:
                label = "Right center punch"
                side_text = "right"

            # Left punch, right chamber
            elif left_is_punch and right_is_chamber:
                label = "Left center punch"
                side_text = "left"

        if label:
            self._record_debug("center_punch_label", label)
            self._add_move_label(label)
            return ("center_punch", side_text)
        
        return None

    def analyze_over_shoulder_punch(self, frame_bgr, landmarks, width, height):
        """
        Detect 'over-the-shoulder' punches and distinguish:
          - same-side (e.g., right arm over right shoulder)
          - cross-body (e.g., right arm over left shoulder)

        Uses:
          - wrist height relative to torso
          - elbow angle (bent or extended, but not ultra-crunched)
          - which shoulder the wrist is closest to in X
          - other hand being clearly lower on the body
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
        r_side_offset = None

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

            # side offset vs body center for inspection
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
        if r_wrist_norm_y is not None and r_side_offset is not None:
            self._debug_line(frame_bgr, "R OS-y", r_wrist_norm_y,)
            self._debug_line(frame_bgr, "R OS-dx", r_side_offset,)

        if l_wrist_norm_y is not None and l_side_offset is not None:
            self._debug_line(frame_bgr, "L OS-y", l_wrist_norm_y,)
            self._debug_line(frame_bgr, "L OS-dx", l_side_offset,)

        # ---- helpers ----
        def _is_over_shoulder_height_angle(wrist_norm_y, elbow_angle) -> bool:
            """High + reasonably open elbow (either bent or extended, not ultra-crunched)."""
            if wrist_norm_y is None or elbow_angle is None:
                return False

            in_height_band = (
                self.OVER_SHOULDER_WRIST_MIN_TORSO_Y
                <= wrist_norm_y
                <= self.OVER_SHOULDER_WRIST_MAX_TORSO_Y
            )
            open_enough = elbow_angle >= self.OVER_SHOULDER_MIN_ELBOW_ANGLE
            return in_height_band and open_enough

        def _other_hand_lower(os_y, other_y, min_diff=0.20) -> bool:
            """
            Check that the blocking/punching wrist is clearly higher on the body
            than the opposite wrist.
            (Remember: smaller torso-y = higher on body.)
            """
            if os_y is None or other_y is None:
                return False
            return (other_y - os_y) >= min_diff

        # Base OS-ness (just height + angle)
        right_os_base = _is_over_shoulder_height_angle(r_wrist_norm_y, r_elbow_angle)
        left_os_base = _is_over_shoulder_height_angle(l_wrist_norm_y, l_elbow_angle)

        labels = []

        # ---- Right arm OS classification ----
        if right_os_base and r_align is not None:
            # same-side: right wrist near right shoulder, clearly higher than left
            if r_align == "right" and _other_hand_lower(r_wrist_norm_y, l_wrist_norm_y):
                labels.append("Right over-shoulder (same)")
            # cross-body: right wrist nearer left shoulder, still higher than left
            elif r_align == "left" and _other_hand_lower(r_wrist_norm_y, l_wrist_norm_y):
                labels.append("Right over-shoulder (cross)")

        # ---- Left arm OS classification ----
        if left_os_base and l_align is not None:
            if l_align == "left" and _other_hand_lower(l_wrist_norm_y, r_wrist_norm_y):
                labels.append("Left over-shoulder (same)")
            elif l_align == "right" and _other_hand_lower(l_wrist_norm_y, r_wrist_norm_y):
                labels.append("Left over-shoulder (cross)")

        if labels:
            os_text = " | ".join(labels)
            self._record_debug("over_shoulder_label", os_text)
            self._add_move_label(os_text)
            
        
        side_text = None
        if any(lbl.startswith("Right") for lbl in labels): side_text = "right"
        if any(lbl.startswith("Left") for lbl in labels): side_text = "left"
        if any(lbl.startswith("Right") for lbl in labels) and any(lbl.startswith("Left") for lbl in labels):
            side_text = "both"
        return ("os_punch", side_text) if labels else None


    # ---------- main run kata method ----------  
    
    def run_kata_sequence(self, frame_bgr, landmarks, frame_count, width, height):
        # always-on stuff (optional)
        self.analyze_horse_stance(frame_bgr, landmarks, frame_count, width, height)

        # if finished, you can idle
        if self._kata_index >= len(self._kata_steps):
            self._add_move_label("Kata complete")
            return

        step = self._kata_steps[self._kata_index]

        # run ONLY this step’s detector
        event = step.detector(frame_bgr, landmarks, width, height)

        # show current step on HUD
        self._add_metric_line("Step", step.name)

        if step.update(event):
            self._add_move_label(f"✔ Completed: {step.name}")
            self._kata_index += 1
            if self._kata_index < len(self._kata_steps):
                self._kata_steps[self._kata_index].reset_runtime()