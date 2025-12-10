import mediapipe as mp
import cv2
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture state
        self.prev_pinch_dist = None
        self.prev_centroid = None
        
        # Output state
        self.state = {
            "is_tracking": False,
            "gesture": "NONE",  # NONE, PAN, ROTATE, ZOOM
            "delta": (0, 0),    # (dx, dy)
            "scale_factor": 1.0
        }
        
        # Smoothing and Hysteresis State
        self.smooth_delta = (0.0, 0.0)
        self.locked_gesture = "NONE"
        self.gesture_lock_frames = 0
        self.alpha_pos = 0.2 # Smoothing factor (Lower = smoother but more lag)
        
        # Hand size tracking for depth-based zoom
        self.prev_hand_size = None
        self.smooth_size_delta = 0.0
        self.zoom_dead_zone = 0.0005
        self.zoom_gain = 3.0
        # Fist detection (relative to initial fist size)
        self.base_fist_dist = None
        self.fist_enter_ratio = 1.2   # <= enters ROTATE
        self.fist_exit_ratio = 1.5    # >= exits ROTATE
        self.fist_calibration_gate = 0.35
        self.fist_calibration_alpha = 0.25

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        self.state = {
            "is_tracking": False,
            "gesture": "NONE",
            "delta": (0, 0),
            "scale_factor": 1.0,
            "landmarks": []
        }

        if results.multi_hand_landmarks:
            self.state["is_tracking"] = True
            self.state["landmarks"] = results.multi_hand_landmarks
            
            # Check for multiple hands
            if len(results.multi_hand_landmarks) == 2:
                self._process_two_hands(results.multi_hand_landmarks)
            else:
                # Use First hand primarily
                self._process_single_hand(results.multi_hand_landmarks[0])
                
        else:
            self.prev_pinch_dist = None
            self.prev_centroid = None
            self.prev_hand_size = None
            self.smooth_size_delta = 0.0
            self.base_fist_dist = None
            self.locked_gesture = "NONE"
            # Decay smooth_delta to zero to prevent residual wobble
            self.smooth_delta = (self.smooth_delta[0] * 0.8, self.smooth_delta[1] * 0.8)
            # Zero out if negligible
            if abs(self.smooth_delta[0]) < 0.0001 and abs(self.smooth_delta[1]) < 0.0001:
                self.smooth_delta = (0.0, 0.0)
            
        return self.state

    def _process_single_hand(self, landmarks):
        # Landmarks
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        
        # 1. Calc Fist-ness (Avg dist of all fingertips to wrist)
        fingertips = [4, 8, 12, 16, 20]
        total_dist_wrist = 0
        for idx in fingertips:
            tip = landmarks.landmark[idx]
            d = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            total_dist_wrist += d
        avg_dist_wrist = total_dist_wrist / 5.0
        # Calibrate fist baseline only when the hand looks closed-ish
        if avg_dist_wrist < self.fist_calibration_gate:
            if self.base_fist_dist is None:
                self.base_fist_dist = max(avg_dist_wrist, 1e-6)
            else:
                self.base_fist_dist = (
                    self.base_fist_dist * (1 - self.fist_calibration_alpha)
                    + avg_dist_wrist * self.fist_calibration_alpha
                )

        norm_fist = avg_dist_wrist / self.base_fist_dist if self.base_fist_dist else None
        
        # 2. Calc Hand Size for depth detection (wrist to middle fingertip)
        # As hand moves TOWARD camera, it appears LARGER
        hand_size = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
        
        # Centroid for movement delta
        coords = np.array([(lm.x, lm.y) for lm in landmarks.landmark])
        centroid = np.mean(coords, axis=0)
        
        # --- State Machine with Hysteresis ---
        # Use fist size relative to the calibrated fist size to decide PAN vs ROTATE
        if norm_fist is None:
            self.locked_gesture = "PAN"
        else:
            if self.locked_gesture == "ROTATE":
                if norm_fist >= self.fist_exit_ratio:
                    self.locked_gesture = "PAN"
            else:
                if norm_fist <= self.fist_enter_ratio:
                    self.locked_gesture = "ROTATE"
                else:
                    self.locked_gesture = "PAN"

        # Debug 
        norm_disp = f"{norm_fist:.2f}" if norm_fist is not None else "NA"
        base_disp = f"{self.base_fist_dist:.3f}" if self.base_fist_dist else "None"
        print(f"Fist: {avg_dist_wrist:.3f} (norm {norm_disp} base {base_disp}) Size: {hand_size:.3f} -> {self.locked_gesture}")     
            
        self.state["gesture"] = self.locked_gesture

        # --- Compute Deltas ---
        if self.prev_centroid is not None:
            dx = centroid[0] - self.prev_centroid[0]
            dy = centroid[1] - self.prev_centroid[1]
            
            # Invert X because camera mirror
            raw_delta = (-dx, dy)
            
            # --- Dead Zone to prevent jitter wobble ---
            dead_zone = 0.005
            magnitude = np.sqrt(raw_delta[0]**2 + raw_delta[1]**2)
            if magnitude < dead_zone:
                raw_delta = (0.0, 0.0)
            
            # --- Smoothing (EMA) ---
            sdx = self.smooth_delta[0] * (1 - self.alpha_pos) + raw_delta[0] * self.alpha_pos
            sdy = self.smooth_delta[1] * (1 - self.alpha_pos) + raw_delta[1] * self.alpha_pos
            self.smooth_delta = (sdx, sdy)
            
            self.state["delta"] = self.smooth_delta
        
        # --- Compute Zoom from Hand Size Change (for ROTATE gesture) ---
        # Legacy: we now use 2-hand zoom. Disable single hand zoom.
        self.state["zoom_delta"] = 0.0
            
        self.prev_centroid = centroid
        self.prev_hand_size = hand_size
            
        self.prev_centroid = centroid
        self.prev_hand_size = hand_size

    def _process_two_hands(self, hands_list):
        # Scale/Zoom based on distance between centroids of two hands
        c1 = np.mean([(lm.x, lm.y) for lm in hands_list[0].landmark], axis=0)
        c2 = np.mean([(lm.x, lm.y) for lm in hands_list[1].landmark], axis=0)
        
        dist = np.sqrt(np.sum((c1 - c2)**2))
        
        if self.prev_pinch_dist is not None:
            scale = dist / self.prev_pinch_dist
            self.state["scale_factor"] = scale
            self.state["gesture"] = "ZOOM"
            print(f"Two Hands: Dist {dist:.1f} -> Scale {scale:.3f} -> ZOOM")
            
        self.prev_pinch_dist = dist
        self.prev_centroid = None # Reset 1-hand state

    def draw_debug(self, frame, landmarks_list):
        if not landmarks_list:
            return
        
        for hand_landmarks in landmarks_list:
            self.mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS
            )
        
        # Draw status
        cv2.putText(frame, f"Gesture: {self.state['gesture']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
