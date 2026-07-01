import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from collections import deque
import threading
import time

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\tanis\OneDrive\Desktop\cv\hand_landmarker.task"

FINGERS = {
    "Thumb":  (4, 2),
    "Index":  (8, 5),
    "Middle": (12, 9),
    "Ring":   (16, 13),
    "Pinky":  (20, 17),
}

FINGER_COLORS = {
    "Thumb":  (255, 80,  80),
    "Index":  (80,  255, 80),
    "Middle": (80,  80,  255),
    "Ring":   (255, 255, 80),
    "Pinky":  (255, 80,  255),
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

JOINT_SETS = [
    (1,2,3),(2,3,4),
    (5,6,7),(6,7,8),
    (9,10,11),(10,11,12),
    (13,14,15),(14,15,16),
    (17,18,19),(18,19,20),
]

TRAIL_LEN = 40

# ── Shared state (thread-safe via lock) ──────────────────────────────────────
latest_result = None
result_lock   = threading.Lock()
trails        = [{name: deque(maxlen=TRAIL_LEN) for name in FINGERS} for _ in range(2)]

# ── Helper functions ──────────────────────────────────────────────────────────
def angle_between(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return int(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def is_up(lm, tip, base):
    return lm[tip].y < lm[base].y

def get_gesture(lm):
    fu = {name: is_up(lm, t, b) for name, (t, b) in FINGERS.items()}
    t, i, m, r, p = fu["Thumb"], fu["Index"], fu["Middle"], fu["Ring"], fu["Pinky"]
    count = sum(fu.values())

    if count == 0:                              return "✊ Fist"
    if count == 5:                              return "🖐 Open Hand"
    if i and not m and not r and not p:         return "☝ Pointing"
    if i and m and not r and not p:             return "✌ Peace"
    if t and not i and not m and not r and not p: return "👍 Thumbs Up"
    if not t and not i and not m and not r and p: return "🤙 Pinky Up"
    if t and p and not i and not m and not r:   return "🤙 Call Me"
    if i and m and r and not p:                 return "🤟 Three Up"
    if i and m and r and p and not t:           return "🖖 Four Up"
    if t and i and not m and not r and not p:   return "👌 Gun"
    if not t and i and not m and not r and p:   return "🤘 Rock On"
    return f"  {count} Fingers Up"

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_hand(frame, lm, hand_idx, h, w):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]

    # Connections
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (220, 220, 220), 2, cv2.LINE_AA)

    # Joints — color by finger group
    finger_joint_map = {
        0: (200, 200, 200),
        **{j: FINGER_COLORS["Thumb"]  for j in [1,2,3,4]},
        **{j: FINGER_COLORS["Index"]  for j in [5,6,7,8]},
        **{j: FINGER_COLORS["Middle"] for j in [9,10,11,12]},
        **{j: FINGER_COLORS["Ring"]   for j in [13,14,15,16]},
        **{j: FINGER_COLORS["Pinky"]  for j in [17,18,19,20]},
    }
    for i, (cx, cy) in enumerate(pts):
        cv2.circle(frame, (cx, cy), 6, finger_joint_map[i], -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(i), (cx + 6, cy - 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.75, (230, 230, 230), 1)

    # Joint angles
    for a, b, c in JOINT_SETS:
        ang = angle_between((lm[a].x, lm[a].y), (lm[b].x, lm[b].y), (lm[c].x, lm[c].y))
        bx, by = pts[b]
        cv2.putText(frame, str(ang), (bx - 12, by - 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 255, 255), 1)

    # Smooth fading trails per fingertip
    trail = trails[hand_idx]
    for name, (tip, _) in FINGERS.items():
        trail[name].append(pts[tip])
        tpts = list(trail[name])
        n = len(tpts)
        for i in range(1, n):
            alpha = int(255 * i / n)
            color = tuple(int(c * alpha / 255) for c in FINGER_COLORS[name])
            thickness = max(1, int(3 * i / n))
            cv2.line(frame, tpts[i-1], tpts[i], color, thickness, cv2.LINE_AA)

    # Finger status bar at bottom
    bar_y1, bar_y2 = h - 65, h - 25
    for fi, (fname, (tip, base)) in enumerate(FINGERS.items()):
        up = is_up(lm, tip, base)
        color = FINGER_COLORS[fname] if up else (50, 50, 50)
        x1, x2 = 10 + fi * 62, 65 + fi * 62
        cv2.rectangle(frame, (x1, bar_y1), (x2, bar_y2), color, -1)
        cv2.rectangle(frame, (x1, bar_y1), (x2, bar_y2), (0,0,0), 1)
        cv2.putText(frame, fname[:3], (x1 + 4, bar_y2 - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 1)

# ── Async callback ────────────────────────────────────────────────────────────
def on_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

# ── Main ──────────────────────────────────────────────────────────────────────
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=on_result
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = time.time()

with HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts = int(time.time() * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector.detect_async(mp_image, ts)

        with result_lock:
            result = latest_result

        if result and result.hand_landmarks:
            for idx, lm in enumerate(result.hand_landmarks):
                draw_hand(frame, lm, idx, h, w)
                gesture    = get_gesture(lm)
                hand_label = "Left" if result.handedness[idx][0].display_name == "Right" else "Right"
                label_color = (0, 255, 120) if hand_label == "Right" else (120, 180, 255)
                cv2.putText(frame, f"{hand_label}: {gesture}", (10, 38 + idx * 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2, cv2.LINE_AA)

        # FPS counter
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv2.putText(frame, "Q = quit", (w - 90, h - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (150, 150, 150), 1)
        cv2.imshow("Finger Motion Detection v2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
