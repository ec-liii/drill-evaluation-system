import cv2
import numpy as np
from ultralytics import YOLO

class DrillPoseEngine:
    def __init__(self):
        self.model = YOLO('yolov8x-pose.pt')
        self.history_buffer = []
        self.buffer_max_size = 60

    def calculate_3pt_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return round(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))), 1)

    def calculate_inclination_angle(self, pt1, pt2):
        delta_x = pt2[0] - pt1[0]
        delta_y = pt2[1] - pt1[1]
        return round(abs(np.degrees(np.arctan2(delta_y, delta_x))), 1)

    def get_velocity_x(self, current_pt, joint_index):
        if len(self.history_buffer) < 5: return 0.0
        past_pt = self.history_buffer[0]['keypoints'][joint_index]
        if past_pt[0] == 0: return 0.0
        return round(abs(current_pt[0] - past_pt[0]) / len(self.history_buffer), 2)

    def detect_cadence(self, keypoint_history, left_idx, right_idx):
        peaks = 0
        for i in range(1, len(keypoint_history) - 1):
            prev_dist = abs(keypoint_history[i-1][left_idx][0] - keypoint_history[i-1][right_idx][0])
            curr_dist = abs(keypoint_history[i][left_idx][0] - keypoint_history[i][right_idx][0])
            next_dist = abs(keypoint_history[i+1][left_idx][0] - keypoint_history[i+1][right_idx][0])
            if curr_dist > prev_dist and curr_dist > next_dist and curr_dist > 15:
                peaks += 1
        if peaks == 0: return 0
        minutes_elapsed = (len(keypoint_history) / 30.0) / 60.0
        return int((peaks * 2) / minutes_elapsed) if minutes_elapsed > 0 else 0

    def evaluate_drill_frame(self, frame, drill_type):
        results = self.model(frame, verbose=False)
        annotated_frame = frame.copy()
        evaluation_log = {"status": "Pass", "metrics": {}, "error_flag": None}

        if not results or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            return annotated_frame, None

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        if len(keypoints) <= 16:
            return annotated_frame, None

        self.history_buffer.append({'keypoints': keypoints})
        if len(self.history_buffer) > self.buffer_max_size: self.history_buffer.pop(0)

        # 1. SALUTE
        if drill_type == "salute":
            r_sh, r_el, r_wr = keypoints[6], keypoints[8], keypoints[10]
            if all(pt[0] > 0 for pt in [r_sh, r_el, r_wr]):
                forearm_angle = self.calculate_inclination_angle(r_el, r_wr)
                wrist_straightness = self.calculate_3pt_angle(r_sh, r_el, r_wr)
                evaluation_log["metrics"] = {"Forearm Inclination": forearm_angle, "Wrist Alignment": wrist_straightness}
                if not (40.0 <= forearm_angle <= 50.0):
                    evaluation_log["status"] = "Fail"
                    evaluation_log["error_flag"] = f"Incorrect Forearm Salute Angle: {forearm_angle}° (Target: 45°)"
                elif wrist_straightness < 165.0:
                    evaluation_log["status"] = "Fail"
                    evaluation_log["error_flag"] = f"Bent Wrist: {wrist_straightness}° (Target: Straight)"

        # 2. KADAM TAL
        elif drill_type == "kadam_tal":
            l_hip, l_knee = keypoints[11], keypoints[13]
            r_hip, r_knee = keypoints[12], keypoints[14]
            
            # FIX: Total insulation validation mapping for hip tracking operations
            if keypoints[5][0] > 0 and keypoints[6][0] > 0 and l_hip[0] > 0 and r_hip[0] > 0:
                l_hip_angle = self.calculate_3pt_angle(keypoints[5], l_hip, l_knee)
                r_hip_angle = self.calculate_3pt_angle(keypoints[6], r_hip, r_knee)
            else:
                l_hip_angle, r_hip_angle = 180.0, 180.0
                
            global_drift = self.get_velocity_x(r_hip, 12)
            evaluation_log["metrics"] = {"Left Hip Flexion": l_hip_angle, "Right Hip Flexion": r_hip_angle}
            if global_drift > 2.5:
                evaluation_log["status"] = "Fail"
                evaluation_log["error_flag"] = "Forward Displacement Drift Detected (Must keep Vx=0)"
            elif l_hip_angle < r_hip_angle and l_hip_angle < 115.0 and not (85.0 <= l_hip_angle <= 105.0):
                evaluation_log["status"] = "Warning"
                evaluation_log["error_flag"] = f"Left Thigh Raised Low: {l_hip_angle}°"
            elif r_hip_angle < l_hip_angle and r_hip_angle < 115.0 and not (85.0 <= r_hip_angle <= 105.0):
                evaluation_log["status"] = "Warning"
                evaluation_log["error_flag"] = f"Right Thigh Raised Low: {r_hip_angle}°"

        # 3. BAJU SWING
        elif drill_type == "bajuswing":
            r_sh, r_el, r_wr = keypoints[6], keypoints[8], keypoints[10]
            l_sh, l_el, l_wr = keypoints[5], keypoints[7], keypoints[9]
            
            if all(pt[0] > 0 for pt in [r_sh, r_el, r_wr, l_sh, l_el, l_wr]):
                r_arm_lock = self.calculate_3pt_angle(r_sh, r_el, r_wr)
                l_arm_lock = self.calculate_3pt_angle(l_sh, l_el, l_wr)
                evaluation_log["metrics"] = {"Right Arm Lock": r_arm_lock, "Left Arm Lock": l_arm_lock}
                if r_arm_lock < 165.0 or l_arm_lock < 165.0:
                    evaluation_log["status"] = "Fail"
                    evaluation_log["error_flag"] = "Bowed/Bent Elbow Joint Detected in Swing"
            else:
                evaluation_log["metrics"] = {"Right Arm Lock": 180.0, "Left Arm Lock": 180.0}

        # 4. TEJ MARCH
        elif drill_type == "tej_march":
            calculated_cadence = self.detect_cadence([f['keypoints'] for f in self.history_buffer], 15, 16)
            evaluation_log["metrics"] = {"Calculated Cadence": f"{calculated_cadence} BPM"}
            if calculated_cadence > 0 and not (110 <= calculated_cadence <= 124):
                evaluation_log["status"] = "Fail"
                evaluation_log["error_flag"] = f"Improper Quick March Pace: {calculated_cadence} BPM (Target: 116-120)"

        # 5. SLOW MARCH
        elif drill_type == "slow_march":
            r_wr = keypoints[10]
            r_arm_velocity = self.get_velocity_x(r_wr, 10)
            calculated_cadence = self.detect_cadence([f['keypoints'] for f in self.history_buffer], 15, 16)
            evaluation_log["metrics"] = {"Arm Wavering Volatility": r_arm_velocity, "Slow Cadence Rate": f"{calculated_cadence} BPM"}
            if r_arm_velocity > 1.2:
                evaluation_log["status"] = "Fail"
                evaluation_log["error_flag"] = "Unsteady Arm Swing Wavering in Slow March"
            elif calculated_cadence > 0 and not (60 <= calculated_cadence <= 75):
                evaluation_log["status"] = "Warning"
                evaluation_log["error_flag"] = f"Improper Tempo: {calculated_cadence} BPM (Target: 65-70)"

        # 6. HILL MARCH
        elif drill_type == "hill_march":
            if keypoints[5][0] > 0 and keypoints[6][0] > 0 and keypoints[11][0] > 0 and keypoints[12][0] > 0:
                mid_sh_x = (keypoints[5][0] + keypoints[6][0]) / 2
                mid_sh_y = (keypoints[5][1] + keypoints[6][1]) / 2
                mid_hp_x = (keypoints[11][0] + keypoints[12][0]) / 2
                mid_hp_y = (keypoints[11][1] + keypoints[12][1]) / 2
                axial_lean = round(abs(90.0 - self.calculate_inclination_angle([mid_hp_x, mid_hp_y], [mid_sh_x, mid_sh_y])), 1)
            else:
                axial_lean = 0.0
                
            evaluation_log["metrics"] = {"Torso Axial Lean": f"{axial_lean}°"}
            if axial_lean < 3.0 and axial_lean > 0.0:
                evaluation_log["status"] = "Warning"
                evaluation_log["error_flag"] = f"Insufficient Center of Gravity Spine Tilt: {axial_lean}°"

        color_theme = (0, 255, 0) if evaluation_log["status"] == "Pass" else ((0, 165, 255) if evaluation_log["status"] == "Warning" else (0, 0, 255))
        cv2.putText(annotated_frame, f"DRILL: {drill_type.upper()} | STATUS: {evaluation_log['status']}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_theme, 2)
        if evaluation_log["error_flag"]:
            cv2.putText(annotated_frame, f"ALERT: {evaluation_log['error_flag']}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return annotated_frame, evaluation_log