import cv2
import numpy as np
import math
import time
from pygame import mixer
import os

# --- DEBUG CONFIG ---
DEBUG_MODE = False  # Set to True to see threshold window and Instrument ROIs

# --- AUDIO INIT ---
try:
    mixer.init()
    # Ensure these files exist in your folder
    SOUND_SNARE = mixer.Sound('caja.mp3')       # Snare (2 fingers)
    SOUND_BASS = mixer.Sound('caja2.mp3')       # Bass (3 fingers)
    SOUND_CYMBAL = mixer.Sound('platillo.mp3')  # Cymbal (4 fingers)
except Exception as e:
    print(f"Error loading sounds: {e}")
    SOUND_SNARE = None
    SOUND_BASS = None
    SOUND_CYMBAL = None

# --- CONSTANTS ---
STATE_WAITING = 0      # Searching for hand (full screen)
STATE_CONFIG = 1       # ROI Locked, selecting instrument/color
STATE_PLAYING = 2      # Playing drums

# Hand Detection
HAND_THRESHOLD = 50
MIN_HAND_AREA = 500
DYNAMIC_DEPTH_RATIO = 0.10

# Timers & Cooldowns
TIME_TO_LOCK_ROI = 2.0         # Time holding 5 fingers to start
TIME_TO_CONFIRM_FINGERS = 1.0  # Time holding a gesture to accept it
CREATION_HOLD_TIME = 1.5       # Time holding stick to create drum
CALIBRATION_TIME = 3.0         # Time to hold color in box to calibrate

# Global Cooldowns
STATE_COOLDOWN_TIME = 3.0      # Safety buffer after changing states
ACTION_COOLDOWN_TIME = 2.0     # Safety buffer after creating instrument or calibrating

# Instrument Types
INSTRUMENT_TYPES = {
    2: {'name': 'SNARE', 'color': (0, 0, 255), 'radius': 80, 'sound': SOUND_SNARE, 'img_file': 'Tambor_pequeno.png'},
    3: {'name': 'BASS',  'color': (0, 255, 0), 'radius': 100, 'sound': SOUND_BASS,  'img_file': 'Tambor_Grande.png'},
    4: {'name': 'CYMBAL','color': (0, 255, 255), 'radius': 80, 'sound': SOUND_CYMBAL,'img_file': 'Platillos.png'}
}

class HandProcessor:
    def __init__(self):
        pass
    
    def calculate_angle(self, start, end, far):
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        if b * c == 0: return 0
        val = (b**2 + c**2 - a**2) / (2 * b * c)
        val = max(-1.0, min(1.0, val))
        return math.degrees(math.acos(val))

    def process(self, img_bgr, offset=(0,0)):
        """
        Returns: count, contour, hull, defect_points, thresh_image
        """
        gx, gy = offset
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, HAND_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, None, None, [], thresh

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < MIN_HAND_AREA:
            return 0, None, None, [], thresh

        _, _, _, h_hand = cv2.boundingRect(c)
        depth_thresh = h_hand * DYNAMIC_DEPTH_RATIO

        hull_idxs = cv2.convexHull(c, returnPoints=False)
        c_global = c + (gx, gy)
        hull_pts_local = cv2.convexHull(c, returnPoints=True)
        hull_global = hull_pts_local + (gx, gy)

        try:
            defects = cv2.convexityDefects(c, hull_idxs)
        except:
            return 1, c_global, hull_global, [], thresh

        if defects is None:
            return 1, c_global, hull_global, [], thresh

        count = 0
        defects_global = []
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(c[s][0])
            end = tuple(c[e][0])
            far = tuple(c[f][0])
            
            depth_val = d / 256.0
            if depth_val > depth_thresh:
                angle = self.calculate_angle(start, end, far)
                if angle < 90:
                    count += 1
                    defects_global.append((far[0] + gx, far[1] + gy))
        
        return count + 1, c_global, hull_global, defects_global, thresh

class StickTracker:
    def __init__(self):
        # HSV Ranges
        self.lower_color = np.array([125, 100, 100], dtype=np.uint8)
        self.upper_color = np.array([165, 255, 255], dtype=np.uint8)
        self.lower_color_2 = None
        self.upper_color_2 = None
        self.circular_range = False
        
        # Stability Logic
        self.last_pos = None
        self.stable_start_time = None
        self.is_stable = False
        self.kernel = np.ones((5, 5), np.uint8)
        self.move_threshold = 30 # px

    def calibrate(self, frame_roi):
        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        mask_valid = hsv[:, :, 1] > 50
        valid_pixels = hsv[mask_valid]
        
        if valid_pixels.size == 0:
            print("Calibration warning: No valid pixels.")
            return False

        h_med = int(np.median(valid_pixels[:, 0]))
        s_med = int(np.median(valid_pixels[:, 1]))
        v_med = int(np.median(valid_pixels[:, 2]))

        margin_h, margin_sv = 20, 60
        h_low, h_high = h_med - margin_h, h_med + margin_h
        s_low = max(0, s_med - margin_sv)
        s_high = min(255, s_med + margin_sv)
        v_low = max(0, v_med - margin_sv)
        v_high = min(255, v_med + margin_sv)

        if h_low < 0:
            self.lower_color = np.array([h_low + 180, s_low, v_low], dtype=np.uint8)
            self.upper_color = np.array([179, s_high, v_high], dtype=np.uint8)
            self.lower_color_2 = np.array([0, s_low, v_low], dtype=np.uint8)
            self.upper_color_2 = np.array([h_high, s_high, v_high], dtype=np.uint8)
            self.circular_range = True
        elif h_high > 179:
            self.lower_color = np.array([h_low, s_low, v_low], dtype=np.uint8)
            self.upper_color = np.array([179, s_high, v_high], dtype=np.uint8)
            self.lower_color_2 = np.array([0, s_low, v_low], dtype=np.uint8)
            self.upper_color_2 = np.array([h_high - 180, s_high, v_high], dtype=np.uint8)
            self.circular_range = True
        else:
            self.lower_color = np.array([h_low, s_low, v_low], dtype=np.uint8)
            self.upper_color = np.array([h_high, s_high, v_high], dtype=np.uint8)
            self.circular_range = False
        
        print(f"Calibration Done. Median: {h_med, s_med, v_med}")
        return True

    def find_stick(self, frame, search_mask=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if self.circular_range:
            mask1 = cv2.inRange(hsv, self.lower_color, self.upper_color)
            mask2 = cv2.inRange(hsv, self.lower_color_2, self.upper_color_2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        if search_mask is not None:
            mask = cv2.bitwise_and(mask, search_mask)

        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 5:
                current_pos = (int(x), int(y))
                self._update_stability(current_pos)
                return current_pos
        
        self.is_stable = False
        self.stable_start_time = None
        self.last_pos = None
        return None

    def _update_stability(self, current_pos):
        if self.last_pos is None:
            self.last_pos = current_pos
            self.stable_start_time = time.time()
            return

        dist = math.sqrt((current_pos[0] - self.last_pos[0])**2 + (current_pos[1] - self.last_pos[1])**2)
        
        if dist < self.move_threshold:
            if time.time() - self.stable_start_time >= CREATION_HOLD_TIME:
                self.is_stable = True
        else:
            self.is_stable = False
            self.stable_start_time = time.time() 
            self.last_pos = current_pos

    def get_stability_progress(self):
        if self.stable_start_time is None: return 0.0
        elapsed = time.time() - self.stable_start_time
        return min(1.0, elapsed / CREATION_HOLD_TIME)

class InstrumentManager:
    def __init__(self):
        self.instruments = [] 
        self.loaded_images = {} # Cache for loaded images
        self._load_images()

    def _load_images(self):
        """Loads and pre-processes images for transparency"""
        for k, v in INSTRUMENT_TYPES.items():
            filename = v['img_file']
            if os.path.exists(filename):
                try:
                    # IMREAD_UNCHANGED is crucial for Alpha Channel in PNGs
                    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Resize to match instrument diameter
                        diameter = v['radius'] * 2
                        img = cv2.resize(img, (diameter, diameter))
                        self.loaded_images[k] = img
                        print(f"Loaded image: {filename}")
                    else:
                        print(f"Failed to load image (None): {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Image not found: {filename}")

    def add_instrument(self, type_id, position):
        info = INSTRUMENT_TYPES.get(type_id)
        if not info: return
        
        r = info['radius']
        
        # Calculate ROI for Optimized Tracking (1.5x size)
        # 1.5 * radius margin from center on all sides
        # ROI width/height = 3 * radius
        margin = int(1.5 * r)
        roi_x = position[0] - margin
        roi_y = position[1] - margin
        roi_w = margin * 2
        roi_h = margin * 2
        
        drum = {
            'pos': position,
            'type': type_id,
            'name': info['name'],
            'radius': r,
            'base_color': info['color'],
            'current_color': info['color'],
            'sound': info['sound'],
            'is_playing': False,
            'image': self.loaded_images.get(type_id),
            'roi': (roi_x, roi_y, roi_w, roi_h) # Store ROI for tracking
        }
        self.instruments.append(drum)
        print(f"Added Instrument: {info['name']} at {position} with ROI {drum['roi']}")

    def clear(self):
        self.instruments = []

    def check_collisions(self, stick_positions):
        """Checks collisions for a list of stick positions."""
        # Convert single position to list for compatibility if needed, 
        # but stick_positions should now be a list.
        if not stick_positions: return

        for drum in self.instruments:
            dx, dy = drum['pos']
            hit = False
            
            # Check if ANY stick hits this drum
            for stick_pos in stick_positions:
                if stick_pos is None: continue
                
                sx, sy = stick_pos
                dist = math.sqrt((sx - dx)**2 + (sy - dy)**2)
                
                if dist < drum['radius']:
                    hit = True
                    break # One hit is enough to trigger
            
            if hit:
                # Trigger sound only on new press (rising edge)
                if not drum['is_playing']:
                    if drum['sound']: drum['sound'].play()
                    drum['is_playing'] = True
            else:
                # Reset when no stick is inside
                drum['is_playing'] = False

    def _overlay_image_alpha(self, img, img_overlay, x, y, alpha_mask):
        """Overlays img_overlay on top of img at (x, y) using alpha_mask."""
        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if outside
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels = img.shape[2]
        # Safety check: ensure mask matches overlay size
        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            # Check overlay channels: if PNG has 4 (BGRA) and img has 3 (BGR), use first 3
            overlay_val = img_overlay[y1o:y2o, x1o:x2o, c] if c < img_overlay.shape[2] else img_overlay[y1o:y2o, x1o:x2o, 0]
            
            img[y1:y2, x1:x2, c] = (alpha * overlay_val +
                                    alpha_inv * img[y1:y2, x1:x2, c])

    def draw(self, frame):
        for drum in self.instruments:
            x, y = drum['pos']
            r = drum['radius']
            
            # 1. Draw Image (if available) - No circles as per request
            if drum['image'] is not None:
                img = drum['image']
                
                # Top-left corner for overlay
                tl_x = x - r
                tl_y = y - r
                
                # Global Transparency Factor
                alpha_factor = 0.6

                # Handle PNG (with Alpha) vs PNG/JPG (no Alpha)
                if img.shape[2] == 4:
                    b, g, r_ch, a = cv2.split(img)
                    overlay_rgb = cv2.merge((b, g, r_ch))
                    mask = (a / 255.0) * alpha_factor
                else:
                    # Treat whole image as semi-transparent
                    overlay_rgb = img
                    h_img, w_img = img.shape[:2]
                    mask = np.ones((h_img, w_img), dtype=np.float32) * alpha_factor
                    
                self._overlay_image_alpha(frame, overlay_rgb, tl_x, tl_y, mask)
            else:
                # Fallback: Draw Circles ONLY if no image is found
                cv2.circle(frame, drum['pos'], r, drum['current_color'], -1)
                cv2.circle(frame, drum['pos'], r, (255, 255, 255), 2)
            
            # 2. Draw Text Label
            cv2.putText(frame, drum['name'], (x-20, y+r+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # 3. DEBUG: Draw ROI Square if enabled
            if DEBUG_MODE:
                # ROI Square for optimization visualization
                roi_x, roi_y, roi_w, roi_h = drum['roi']
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 255), 1)

class DrumApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 960)
        
        self.state = STATE_WAITING
        self.hand_detector = HandProcessor()
        
        # 2 Independent Stick Trackers
        self.sticks = [StickTracker(), StickTracker()] 
        self.next_stick_to_calibrate = 0 # 0 for Stick 1, 1 for Stick 2

        self.instruments = InstrumentManager()
        
        self.roi = None # (x, y, w, h)
        self.roi_padding = 40
        self.selected_instrument_type = None
        self.calibrating_mode = False
        
        # Timer variables
        self.waiting_timer_start = None
        
        # Cooldown Timestamps
        self.last_state_change_time = 0
        self.last_action_time = 0  
        
        # Finger stability logic
        self.stable_finger_count = 0  
        self.pending_finger_count = 0 
        self.finger_stable_start = None
        
        # Debug / Keyboard Simulation
        self.simulated_fingers = 0

        # Calibration specific timer
        self.calibration_timer_start = None
        self.show_calib_success_msg = 0

    def _change_state(self, new_state):
        self.state = new_state
        self.last_state_change_time = time.time()
        # Reset finger tracking on state change
        self.stable_finger_count = 0
        self.pending_finger_count = 0
        self.finger_stable_start = None
        # FIX: Reset simulated fingers to avoid auto-triggering next state
        self.simulated_fingers = 0 

    def _trigger_action_cooldown(self):
        """Called after a successful create/calibrate to prevent double actions"""
        self.last_action_time = time.time()
        # Reset stick stability to force user to move and stop again
        for s in self.sticks:
            s.is_stable = False
            s.stable_start_time = time.time()

    def _reset_finger_states(self):
        """Forces the finger state to reset to 0 (No selection)"""
        self.stable_finger_count = 0
        self.pending_finger_count = 0
        self.finger_stable_start = None
        self.selected_instrument_type = None
        # FIX: Reset simulated fingers to avoid repeated actions
        self.simulated_fingers = 0

    def _update_finger_stability(self, raw_count):
        # CHANGE: Ignore 0 fingers. If hand is lost, keep the last confirmed state.
        if raw_count == 0:
            self.finger_stable_start = None
            self.pending_finger_count = 0
            return

        if raw_count == self.pending_finger_count:
            if self.finger_stable_start is None:
                self.finger_stable_start = time.time()
            else:
                elapsed = time.time() - self.finger_stable_start
                if elapsed >= TIME_TO_CONFIRM_FINGERS:
                    self.stable_finger_count = raw_count
        else:
            self.pending_finger_count = raw_count
            self.finger_stable_start = time.time()

    def _draw_hand_debug(self, frame, contour, hull, defects):
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        if hull is not None:
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), 1)
        for p in defects:
            cv2.circle(frame, p, 5, (255, 0, 0), -1)

    def _track_sticks_in_rois(self, frame):
        """
        Optimized stick tracking for MULTIPLE sticks.
        Returns a list of positions: [pos_stick_1, pos_stick_2]
        """
        stick_positions = [None, None]

        # For each stick tracker
        for i, stick_tracker in enumerate(self.sticks):
            # Check all instrument ROIs to find this stick
            # Note: A stick can only be in one place, so if found in one ROI, 
            # we can technically stop searching for THAT stick in other ROIs to save time.
            # But overlapping ROIs might make this tricky. Let's do simple search first.
            
            for drum in self.instruments.instruments:
                rx, ry, rw, rh = drum['roi']
                
                # Clamp to frame
                h_img, w_img = frame.shape[:2]
                x1 = max(0, rx)
                y1 = max(0, ry)
                x2 = min(w_img, rx + rw)
                y2 = min(h_img, ry + rh)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                roi_frame = frame[y1:y2, x1:x2]
                
                local_pos = stick_tracker.find_stick(roi_frame)
                
                if local_pos:
                    # Convert to global
                    global_pos = (local_pos[0] + x1, local_pos[1] + y1)
                    stick_positions[i] = global_pos
                    break # Found this stick, stop checking other ROIs for this stick
        
        return stick_positions

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # KEYBOARD INPUT HANDLING
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break # ESC to exit

            # Keyboard Debug: Simulate Fingers
            if key in [ord(str(i)) for i in range(6)]: # Keys 0-5
                self.simulated_fingers = int(chr(key))
                print(f"DEBUG: Simulated Fingers set to {self.simulated_fingers}")

            # Keyboard Debug: Force State
            if key == ord('w'):
                self._change_state(STATE_WAITING)
                print("DEBUG: Forced state WAITING")
            elif key == ord('c'):
                if self.roi is None:
                    h, w = frame.shape[:2]
                    self.roi = (int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5))
                self._change_state(STATE_CONFIG)
                print("DEBUG: Forced state CONFIG")
            elif key == ord('p'):
                self._change_state(STATE_PLAYING)
                print("DEBUG: Forced state PLAYING")

            # 1. Global Processing
            if DEBUG_MODE:
                f_count = self.simulated_fingers
                f_cont, f_hull, f_defects = None, None, []
                thresh_full = None
            else:
                f_count, f_cont, f_hull, f_defects, thresh_full = self.hand_detector.process(frame)
            
            effective_fingers_global = f_count
            
            if DEBUG_MODE and thresh_full is not None:
                cv2.imshow("Debug - Hand Threshold", thresh_full)
            
            # Stick Tracking Strategy
            stick_positions_global = [None, None]
            
            if self.state == STATE_PLAYING:
                 # Optimization: Only track inside Instrument ROIs
                 stick_positions_global = self._track_sticks_in_rois(frame)
            else:
                 # Full screen tracking for Setup/Config
                 stick_positions_global[0] = self.sticks[0].find_stick(frame)
                 stick_positions_global[1] = self.sticks[1].find_stick(frame)
            
            # Draw Stick Pointers
            colors = [(0, 255, 255), (255, 0, 255)] # Stick 1: Yellow/Cyan, Stick 2: Magenta
            for i, pos in enumerate(stick_positions_global):
                if pos:
                    cv2.circle(display_frame, pos, 10, colors[i], 2)
                    cv2.circle(display_frame, pos, 3, (0, 0, 0), -1)
                    cv2.putText(display_frame, f"S{i+1}", (pos[0]+10, pos[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

            # --- STATE MACHINE ---
            if self.state == STATE_WAITING:
                self._handle_waiting(display_frame, effective_fingers_global, f_cont, f_hull, f_defects)
            
            elif self.state == STATE_CONFIG:
                self._handle_config(frame, display_frame, stick_positions_global, override_fingers=self.simulated_fingers)

            elif self.state == STATE_PLAYING:
                self._handle_playing(frame, display_frame, effective_fingers_global, stick_positions_global)

            # Draw Instruments
            self.instruments.draw(display_frame)

            # UI Status
            state_text = ["WAITING", "CONFIG", "PLAYING"][self.state]
            
            # Messages
            if time.time() - self.show_calib_success_msg < 2.0:
                 # Show which stick was calibrated
                 calibrated_stick_id = 1 if self.next_stick_to_calibrate == 0 else 2 # The one just finished is prev
                 cv2.putText(display_frame, f"STICK {calibrated_stick_id} CALIBRATED!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            cv2.putText(display_frame, f"STATE: {state_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.simulated_fingers > 0 or DEBUG_MODE:
                cv2.putText(display_frame, f"DEBUG FINGERS: {self.simulated_fingers}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Virtual Drums", display_frame)

        self.cap.release()
        cv2.destroyAllWindows()

    def _handle_waiting(self, display, fingers, contour, hull, defects):
        self._draw_hand_debug(display, contour, hull, defects)
        
        if time.time() - self.last_state_change_time < STATE_COOLDOWN_TIME:
            return

        if fingers == 5:
            if self.waiting_timer_start is None:
                self.waiting_timer_start = time.time()
            
            elapsed = time.time() - self.waiting_timer_start
            cv2.putText(display, f"Hold to Start: {elapsed:.1f}/{TIME_TO_LOCK_ROI}s", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if elapsed >= TIME_TO_LOCK_ROI:
                if contour is not None:
                    x, y, w, h = cv2.boundingRect(contour)
                    x = max(0, x - self.roi_padding)
                    y = max(0, y - self.roi_padding)
                    w = min(display.shape[1] - x, w + 2*self.roi_padding)
                    h = min(display.shape[0] - y, h + 2*self.roi_padding)
                    self.roi = (x, y, w, h)
                elif DEBUG_MODE:
                    h, w = display.shape[:2]
                    self.roi = (int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5))
                
                if self.roi is not None:
                    self._change_state(STATE_CONFIG)
                    self.waiting_timer_start = None
                    print(f"State Changed: CONFIG. ROI: {self.roi}")
        else:
            self.waiting_timer_start = None
            cv2.putText(display, "Show 5 Fingers to Start", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def _handle_config(self, frame, display, stick_positions, override_fingers=0):
        # NOTE: stick_positions is a list [pos1, pos2]
        
        rx, ry, rw, rh = self.roi
        cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
        
        if DEBUG_MODE:
             f_roi = override_fingers
             c_roi, h_roi, d_roi = None, None, []
        else:
             roi_img = frame[ry:ry+rh, rx:rx+rw]
             f_roi, c_roi, h_roi, d_roi, _ = self.hand_detector.process(roi_img, offset=(rx, ry))
             if override_fingers > 0:
                 f_roi = override_fingers

        self._draw_hand_debug(display, c_roi, h_roi, d_roi)
        self._update_finger_stability(f_roi)
        
        if self.finger_stable_start is not None:
            hold_time = time.time() - self.finger_stable_start
            if hold_time < TIME_TO_CONFIRM_FINGERS:
                cv2.putText(display, f"Detecting: {f_roi} ({hold_time:.1f}s)", (rx, ry-35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(display, f"Stable: {self.stable_finger_count}", 
                    (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if time.time() - self.last_action_time < ACTION_COOLDOWN_TIME:
            wait = ACTION_COOLDOWN_TIME - (time.time() - self.last_action_time)
            cv2.putText(display, f"Action Cooldown: {wait:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        # 3. Logic based on STABLE count
        
        # --- CALIBRATION (1 Finger) ---
        if self.stable_finger_count == 1:
            h_scr, w_scr, _ = frame.shape
            box_w, box_h = 40, 40
            bx1 = (w_scr // 2) - (box_w // 2)
            by1 = 20
            bx2 = bx1 + box_w
            by2 = by1 + box_h
            
            # Identify which stick is being calibrated
            current_stick_id = self.next_stick_to_calibrate + 1
            color_text = (0, 255, 255) if current_stick_id == 1 else (255, 0, 255)
            
            cv2.rectangle(display, (bx1, by1), (bx2, by2), color_text, 2)
            cv2.putText(display, f"Place Stick {current_stick_id} Here", (bx1-40, by2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)

            if self.calibration_timer_start is None:
                self.calibration_timer_start = time.time()
            
            calib_elapsed = time.time() - self.calibration_timer_start
            remaining = max(0.0, CALIBRATION_TIME - calib_elapsed)
            
            cv2.putText(display, f"Calibrating: {remaining:.1f}s", (bx1-40, by2+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if calib_elapsed >= CALIBRATION_TIME:
                calib_roi = frame[by1:by2, bx1:bx2]
                
                # Calibrate the specific stick tracker
                tracker_to_calibrate = self.sticks[self.next_stick_to_calibrate]
                success = tracker_to_calibrate.calibrate(calib_roi)
                
                if success:
                    self.show_calib_success_msg = time.time()
                    self._trigger_action_cooldown() 
                    self._reset_finger_states()
                    # Toggle stick for next time
                    self.next_stick_to_calibrate = (self.next_stick_to_calibrate + 1) % 2
                self.calibration_timer_start = None 
        else:
            self.calibration_timer_start = None 

        # --- INSTRUMENT SELECTION (2, 3, 4 Fingers) ---
        if self.stable_finger_count in [2, 3, 4]:
            self.calibrating_mode = False
            self.selected_instrument_type = self.stable_finger_count
            name = INSTRUMENT_TYPES[self.stable_finger_count]['name']
            cv2.putText(display, f"Selected: {name} (Hold Stick 1)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- EXIT TO PLAYING (5 Fingers) ---
        state_elapsed = time.time() - self.last_state_change_time
        if self.stable_finger_count == 5:
            if state_elapsed >= STATE_COOLDOWN_TIME:
                self._change_state(STATE_PLAYING)
                print("State Changed: PLAYING")
                return
            else:
                wait_time = STATE_COOLDOWN_TIME - state_elapsed
                cv2.putText(display, f"Exit locked: {wait_time:.1f}s", (rx, ry+rh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 4. Stick Logic (Creation) - ONLY WITH STICK 1
        stick1_pos = stick_positions[0]
        stick1_tracker = self.sticks[0]

        if self.selected_instrument_type and self.stable_finger_count != 1:
            if stick1_pos:
                progress = stick1_tracker.get_stability_progress()
                radius = INSTRUMENT_TYPES[self.selected_instrument_type]['radius']
                
                # Ghost instrument
                cv2.circle(display, stick1_pos, radius, (100, 100, 100), 1)
                # Filling animation
                fill_rad = int(radius * progress)
                cv2.circle(display, stick1_pos, fill_rad, (0, 255, 255), -1)

                if stick1_tracker.is_stable:
                    self.instruments.add_instrument(self.selected_instrument_type, stick1_pos)
                    self._trigger_action_cooldown() 
                    self._reset_finger_states()     

    def _handle_playing(self, frame, display, fingers_global, stick_positions):
        # 1. Reset Check (Global 5 fingers)
        if time.time() - self.last_state_change_time > STATE_COOLDOWN_TIME:
            if fingers_global == 5:
                if not DEBUG_MODE:
                    _, c_global, _, _, _ = self.hand_detector.process(frame)
                    if c_global is not None:
                        x, y, w, h = cv2.boundingRect(c_global)
                        x = max(0, x - self.roi_padding)
                        y = max(0, y - self.roi_padding)
                        w = min(frame.shape[1] - x, w + 2*self.roi_padding)
                        h = min(frame.shape[0] - y, h + 2*self.roi_padding)
                        self.roi = (x, y, w, h)
                elif self.roi is None:
                     h, w = frame.shape[:2]
                     self.roi = (int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5))

                self.instruments.clear()
                self._change_state(STATE_CONFIG)
                print("Reset triggered. Back to CONFIG.")
                return

        # 2. Track Stick Collisions (Multi-stick)
        self.instruments.check_collisions(stick_positions)

if __name__ == "__main__":
    app = DrumApp()
    app.run()
