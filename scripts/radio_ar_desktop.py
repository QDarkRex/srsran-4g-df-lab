import cv2
import os
import socket
import math
import time
import statistics
import numpy as np
import argparse
import json
import threading
from collections import deque

# --- 0. ARGUMENT PARSING ---
import urllib.request
import urllib.parse

parser = argparse.ArgumentParser(description="Tactical Radio Tracker")
parser.add_argument("--postfix", type=str, default="", help="Only show IMSIs ending with this postfix")
parser.add_argument("--target-imsi", type=str, default="", help="Target IMSI to monitor")
parser.add_argument("--antenna-spacing-cm", type=float, default=None,
                    help="Measured center-to-center RX0/RX1 spacing in cm. If omitted, "
                         "assumes exactly half-wavelength (lambda/2). WRONG value here "
                         "biases every AoA reading — measure the real spacing.")
parser.add_argument("--max-aoa-jump-deg", type=float, default=35.0,
                    help="Reject/freeze a sample if AoA jumps more than this many degrees "
                         "from the last good sample (kills phase-wrap edge-snapping). "
                         "A real target cannot cross the field of view in one TTI.")
parser.add_argument("--endfire-limit-deg", type=float, default=60.0,
                    help="Flag readings beyond +/- this angle as low-confidence: a 2-element "
                         "interferometer is unreliable near +/-90deg (endfire).")
parser.add_argument("--aoa-median-window", type=int, default=5,
                    help="Number of recent AoA samples to median-filter over (rejects single "
                         "outlier snaps). Set to 1 to disable.")
parser.add_argument("--cam-hfov", type=float, default=90.0,
                    help="Measured horizontal field of view of the camera lens, in degrees. "
                         "The HUD maps AoA degrees linearly onto pixel position assuming the "
                         "frame center is broadside (0deg) and the frame edges are +/-HFOV/2 "
                         "— a wrong value here makes every marker position wrong even if the "
                         "AoA itself is correct. Default 90 is a guess, not a measurement; see "
                         "TESTING_CHECKLIST.md for how to measure your actual lens HFOV. Can "
                         "still be nudged live with the 'w'/'s' keys once running.")
args_parsed = parser.parse_args()
POSTFIX = args_parsed.postfix
TARGET_IMSI = args_parsed.target_imsi
MAX_AOA_JUMP_DEG = args_parsed.max_aoa_jump_deg
ENDFIRE_LIMIT_DEG = args_parsed.endfire_limit_deg
AOA_MEDIAN_WINDOW = max(1, args_parsed.aoa_median_window)

# Telegram credentials, read from the environment (never hardcode a live bot
# token in source — a previous token was committed here and had to be
# revoked). Set these before running:
#   export TG_BOT_TOKEN=...
#   export TG_CHAT_ID=...
# If unset, Telegram notifications are just skipped (see send_telegram_notification).
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
if not TG_TOKEN or not TG_CHAT_ID:
    print("[Telegram] TG_BOT_TOKEN / TG_CHAT_ID not set in environment — notifications disabled.")

notification_armed = False

def send_telegram_notification(msg):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[Telegram] Config missing. Message: {msg}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        data = urllib.parse.urlencode({"chat_id": TG_CHAT_ID, "text": msg}).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=5) as response:
            pass
        print(f"[Telegram] Notification sent: {msg}")
    except Exception as e:
        print(f"[Telegram] Failed to send notification: {e}")

def telegram_polling_thread():
    global notification_armed
    if not TG_TOKEN:
        return

    last_update_id = 0
    # Discard existing updates to avoid retroactively triggering from old messages
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode())
            if res.get("ok") and res.get("result"):
                last_update_id = res["result"][-1]["update_id"] + 1
    except Exception as e:
        print(f"[Telegram Poller] Init error: {e}")

    backoff = 1  # seconds, doubles on consecutive errors, resets on success
    while True:
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates?offset={last_update_id}&timeout=10"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as response:
                res = json.loads(response.read().decode())
                if res.get("ok") and res.get("result"):
                    for update in res["result"]:
                        last_update_id = update["update_id"] + 1
                        message = update.get("message", {})
                        text = message.get("text", "")
                        chat_id = str(message.get("chat", {}).get("id", ""))

                        if chat_id == TG_CHAT_ID and text.strip().startswith("/target"):
                            notification_armed = True
                            send_telegram_notification("Target notification armed. Awaiting next detection.")
                backoff = 1  # reset on success
        except Exception as e:
            print(f"[Telegram Poller] Error: {e} — retrying in {backoff}s")
            time.sleep(min(backoff, 60))  # cap at 1 minute
            backoff = min(backoff * 2, 60)

# Start background Telegram update poller
threading.Thread(target=telegram_polling_thread, daemon=True).start()

# --- 1. RADIO CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5555

# Global EARFCN setting - change this to your desired value
# MUST match srsenb/enb.conf's dl_earfcn. This script has no way to detect a
# mismatch — a wrong value here silently biases LAMBDA/D and every AoA reading,
# with no error. Check the two are equal after any config change.
EARFCN = 1455  # Default Band 1


def _read_earfcn_from_enb_conf(conf_path):
    """Try to read dl_earfcn from srsRAN's enb.conf.  Returns int or None."""
    try:
        with open(conf_path, "r") as f:
            in_rf = False
            for line in f:
                stripped = line.strip()
                if stripped.startswith("[rf]"):
                    in_rf = True
                elif stripped.startswith("[") and in_rf:
                    break  # left the [rf] section
                elif in_rf and "dl_earfcn" in stripped:
                    # lines like:  dl_earfcn = 1455
                    val = stripped.split("=", 1)[1].strip()
                    return int(val)
    except Exception:
        pass
    return None


# Auto-validate EARFCN against enb.conf if it can be found
_enb_conf_paths = [
    os.path.join(os.path.dirname(__file__), "..", "srsenb", "enb.conf"),
    os.path.join(os.path.dirname(__file__), "..", "srsenb", "enb.conf.example"),
]
for _p in _enb_conf_paths:
    _conf_earfcn = _read_earfcn_from_enb_conf(_p)
    if _conf_earfcn is not None:
        if _conf_earfcn != EARFCN:
            print(f"\n⚠️  WARNING: EARFCN mismatch!")
            print(f"   Script EARFCN = {EARFCN}, enb.conf dl_earfcn = {_conf_earfcn}")
            print(f"   AoA readings will be WRONG.  Updating EARFCN to {_conf_earfcn}.")
            EARFCN = _conf_earfcn
        break

# Calculate frequency based on EARFCN
# For Band 1 (2100 MHz): EARFCN 0-599 maps to 1920-1980 MHz UL
# Formula: F_UL = 1920 + 0.1 * (EARFCN)  (for Band 1)
def earfcn_to_frequency(earfcn):
    # Band 1 (2100 MHz) mapping
    if 0 <= earfcn <= 599:
        return (1920.0 + 0.1 * earfcn) * 1e6  # Convert to Hz
    # Band 3 (1800 MHz) example - add more bands as needed
    elif 1200 <= earfcn <= 1949:
        return (1710.0 + 0.1 * (earfcn - 1200)) * 1e6
    # Band 7 (2600 MHz)
    elif 2750 <= earfcn <= 3449:
        return (2500.0 + 0.1 * (earfcn - 2750)) * 1e6
    # Default fallback
    else:
        return (1920.0 + 0.1 * earfcn) * 1e6

FREQ_UL = earfcn_to_frequency(EARFCN)
C = 299792458
LAMBDA = C / FREQ_UL

# Antenna spacing. The AoA math (ratio = LAMBDA*phi / (2*pi*D)) is only correct
# if D matches the REAL physical center-to-center RX0/RX1 spacing. Half-wavelength
# is the ideal (spans exactly +/-90deg over the +/-pi phase range with no
# ambiguity), but if the hardware isn't exactly lambda/2 the readings are biased
# and phase wrapping becomes more frequent — so allow a measured override.
LAMBDA_HALF = LAMBDA * 0.5
if args_parsed.antenna_spacing_cm is not None:
    D = args_parsed.antenna_spacing_cm / 100.0
else:
    D = LAMBDA_HALF

print(f"EARFCN: {EARFCN}")
print(f"Frequency: {FREQ_UL / 1e6:.2f} MHz")
print(f"Wavelength: {LAMBDA * 100:.2f} cm  (lambda/2 = {LAMBDA_HALF * 100:.2f} cm)")
print(f"Antenna spacing (D): {D * 100:.2f} cm"
      + ("  [measured]" if args_parsed.antenna_spacing_cm is not None else "  [assumed lambda/2]"))
if D > LAMBDA_HALF * 1.05:
    print(f"⚠️  WARNING: spacing {D*100:.2f} cm > lambda/2 ({LAMBDA_HALF*100:.2f} cm): "
          f"expect phase ambiguity / edge-snapping toward the field-of-view edges.")

# --- 2. CALIBRATION & VISUALS ---
# IMPORTANT: this is a SECOND, independent phase-offset knob on top of
# /tmp/df_calibration.conf on the eNB side (chest_ul.c). spatial_delta
# arriving over UDP is already offset-corrected if that file exists. If you
# ALSO tune PHASE_CORRECTION here, you are stacking two corrections and will
# not be able to tell which one is responsible for what you see.
#
# Recommended workflow: keep this at 0.0 and do the real calibration once on
# the eNB side with calibrate_df.py -> /tmp/df_calibration.conf (restart
# srsenb after). Only use the 'a'/'d' live-tuning keys here as a quick visual
# sanity check (point the reference at true broadside, tap a/d until the HUD
# reads ~0), then throw that number away — don't leave it applied here AND
# in df_calibration.conf at the same time.
CAM_HFOV = args_parsed.cam_hfov
PHASE_CORRECTION = 0.0
INVERT_DIRECTION = False

print(f"Camera HFOV: {CAM_HFOV:.1f} deg"
      + ("  [default guess — measure the real lens HFOV, see TESTING_CHECKLIST.md]"
         if args_parsed.cam_hfov == 90.0 else "  [provided]"))

GAUSSIAN_SIGMA = 100         # Adjusted for 720p
HEATMAP_ALPHA_PEAK = 0.6

# --- 3. INITIALIZE STATE ---
# active_targets will store a dictionary of targets:
# active_targets = {
#     imsi_str: {
#         'current_aoa': float,
#         'current_mag': float,
#         'smooth_x': float,
#         'smooth_mag': float,
#         'last_seen': float
#     }
# }
active_targets = {}
# Global state for tracking Telegram detection/detachment timeouts separately from GUI timeout
# detection_states = { imsi: { 'state': 'detected'|'detached', 'last_seen': timestamp, 'notified_detection': bool } }
detection_states = {}

# --- 4. HUD FUNCTIONS ---
def draw_tactical_compass(img, targets):
    h, w = img.shape[:2]
    # Position: Bottom Right (scaled for 720p)
    cx, cy = w - 150, h - 150
    r = 100
    
    # 1. Glass HUD Background
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r, (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # 2. Outer Ring & Ticks
    cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
    for a in range(0, 360, 10):
        rad = math.radians(a - 90)
        tick_len = 15 if a % 30 == 0 else 8
        x1 = int(cx + (r - tick_len) * math.cos(rad))
        y1 = int(cy + (r - tick_len) * math.sin(rad))
        x2 = int(cx + r * math.cos(rad))
        y2 = int(cy + r * math.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 3. Heading Needle for each active target
    for imsi, target in targets.items():
        # Skip if IMSI contains "RNTI" (case insensitive)
        if "RNTI" in imsi.upper():
            continue
            
        angle = target['current_aoa']
        magnitude = target['smooth_mag']
        target_rad = math.radians(angle - 90)
        color = (0, 255, 0)
        if magnitude < 5 or target.get('low_confidence'): color = (0, 150, 255) # Warning Amber
        
        tip_x = int(cx + (r - 10) * math.cos(target_rad))
        tip_y = int(cy + (r - 10) * math.sin(target_rad))
        cv2.line(img, (cx, cy), (tip_x, tip_y), color, 3)
        cv2.circle(img, (tip_x, tip_y), 8, color, -1)

        # Label needle with last 4 digits of IMSI
        label = imsi[-4:] if len(imsi) >= 4 else imsi
        cv2.putText(img, label, (tip_x - 15, tip_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 4. Heading Readout
    cv2.putText(img, "COMPASS", (cx - 35, cy + r + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# --- 5. SETUP WINDOW & VIDEO ---
window_name = "Tactical Radio Tracker"
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
# SET RESOLUTION TO 720p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- 6. SETUP NETWORK ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

print("--- 720p HUD ACTIVE ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)  # 1 = horizontal flip, 0 = vertical, -1 = both
    h, w, _ = frame.shape
    center_x = w // 2

    # --- 7. PROCESS RADIO DATA ---
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            msg = data.decode().split(',')
            # Wire format: imsi,delta,mag[,csi_var]. csi_var is the same
            # multipath/quality indicator the console dashboard on the eNB
            # side labels LOW/MID/HIGH - older senders may not send it.
            csi_var = 0.0
            if len(msg) >= 4:
                imsi = msg[0]
                raw_delta_phi = float(msg[1])
                mag = float(msg[2])
                csi_var = float(msg[3])
            elif len(msg) == 3:
                imsi = msg[0]
                raw_delta_phi = float(msg[1])
                mag = float(msg[2])
            elif len(msg) == 2:
                imsi = "UNKNOWN"
                raw_delta_phi = float(msg[0])
                mag = float(msg[1])
            else:
                continue

            # Skip processing if IMSI contains "RNTI" (case insensitive)
            if "RNTI" in imsi.upper():
                continue

            # Filter by postfix if specified
            if POSTFIX and not imsi.endswith(POSTFIX):
                continue

            calibrated_phi = raw_delta_phi - PHASE_CORRECTION
            while calibrated_phi > math.pi: calibrated_phi -= 2 * math.pi
            while calibrated_phi < -math.pi: calibrated_phi += 2 * math.pi

            ratio = (LAMBDA * calibrated_phi) / (2 * math.pi * D)
            clamped_ratio = max(-1.0, min(1.0, ratio))
            aoa = math.degrees(math.asin(clamped_ratio))

            if INVERT_DIRECTION: aoa *= -1

            # A sample is untrustworthy if the phase implies a physically
            # impossible angle (|ratio| > 1 before clamping - asin() would
            # have silently snapped the marker to +/-90deg), if the
            # multipath/quality indicator is HIGH (same 0.40 threshold used
            # by the console dashboard), or if the angle is out in the endfire
            # zone where a 2-element interferometer is inherently unreliable
            # (#4). This is the fix for "camera jitters once the amplifier is
            # attached": a noisy/clipped RX chain producing brief bad readings
            # that got plotted at face value, including hard snaps to the edge.
            low_confidence = (abs(ratio) > 1.0) or (csi_var > 0.40) or (abs(aoa) > ENDFIRE_LIMIT_DEG)

            # Update target state
            now = time.time()
            if imsi not in active_targets:
                active_targets[imsi] = {
                    'current_aoa': aoa,
                    'current_mag': mag,
                    'smooth_x': center_x,
                    'smooth_mag': mag,
                    'last_seen': now,
                    'low_confidence': low_confidence,
                    'aoa_hist': deque([aoa], maxlen=AOA_MEDIAN_WINDOW)
                }
                if not TARGET_IMSI or imsi == TARGET_IMSI:
                    if notification_armed:
                        notification_armed = False
                        if POSTFIX:
                            send_telegram_notification(f"Target IMSI {imsi} (postfix: {POSTFIX}) detected")
                        else:
                            send_telegram_notification(f"IMSI {imsi} detected")
            else:
                target = active_targets[imsi]

                # #1 Temporal continuity guard: a real device cannot cross the
                # field of view in one ~1ms TTI. If the new angle jumps more
                # than MAX_AOA_JUMP_DEG from the last accepted angle, it's a
                # phase-wrap artifact (the +pi/-pi boundary flips the sign and
                # teleports the marker edge-to-edge). Note the low_confidence
                # gate above does NOT catch this, because a wrapped phase is
                # still a "valid" ratio in [-1, 1] — it just points the wrong
                # way. Freeze the angle on such a jump instead of plotting it.
                aoa_jump = abs(aoa - target['current_aoa'])
                wrapped_jump = aoa_jump > MAX_AOA_JUMP_DEG

                if not low_confidence and not wrapped_jump:
                    # #2 Median filter on the ANGLE (not just the pixel), to
                    # reject single-sample outliers that slipped through.
                    target['aoa_hist'].append(aoa)
                    target['current_aoa'] = statistics.median(target['aoa_hist'])

                target['current_mag'] = mag
                target['last_seen'] = now
                target['low_confidence'] = low_confidence or wrapped_jump
                
        except BlockingIOError:
            break
        except Exception:
            break

    # --- 8. DYNAMIC MAPPING & TIMEOUTS ---
    now = time.time()
    stale_keys = []
    for imsi, target in active_targets.items():
        # Skip if IMSI contains "RNTI" (case insensitive)
        if "RNTI" in imsi.upper():
            stale_keys.append(imsi)
            continue
            
        # Hide/remove from GUI if not updated in 2.0 seconds
        if now - target['last_seen'] > 2.0:
            stale_keys.append(imsi)
            continue
            
        target['smooth_mag'] = 0.85 * target['smooth_mag'] + 0.15 * target['current_mag']
        
        target_x_offset = (target['current_aoa'] / (CAM_HFOV / 2)) * center_x
        target_x = int(center_x + target_x_offset)
        target['smooth_x'] = int(0.85 * target['smooth_x'] + 0.15 * target_x)
        target['smooth_x'] = max(0, min(w, target['smooth_x']))

    for k in stale_keys:
        if k in active_targets:  # Check if still exists
            del active_targets[k]

    # Process separate 10.0 second grace period for Telegram Detachment
    # 1. Update/Add current active targets in detection_states
    for imsi, target in active_targets.items():
        if imsi not in detection_states:
            detection_states[imsi] = {
                'state': 'detected',
                'last_seen': target['last_seen']
            }
        else:
            detection_states[imsi]['last_seen'] = target['last_seen']
            if detection_states[imsi]['state'] == 'detached':
                detection_states[imsi]['state'] = 'detected'

    # 2. Check for timeouts (10 seconds) on tracked states
    detached_to_notify = []
    for imsi, state_info in list(detection_states.items()):
        if now - state_info['last_seen'] > 10.0:
            if state_info['state'] == 'detected':
                state_info['state'] = 'detached'
                if not TARGET_IMSI or imsi == TARGET_IMSI:
                    detached_to_notify.append(imsi)

    for k in detached_to_notify:
        if POSTFIX:
            send_telegram_notification(f"Target IMSI {k} (postfix: {POSTFIX}) detached")
        else:
            send_telegram_notification(f"IMSI {k} detached")

    # --- 9. RENDER HEATMAP ---
    alpha_map = np.zeros(w, dtype=float)
    for imsi, target in active_targets.items():
        # Skip if IMSI contains "RNTI" (case insensitive)
        if "RNTI" in imsi.upper():
            continue
            
        dynamic_alpha = np.clip((target['smooth_mag'] / 15.0) * HEATMAP_ALPHA_PEAK, 0.1, 0.8)
        dynamic_sigma = int(np.clip(target['smooth_mag'] * 10, 40, GAUSSIAN_SIGMA * 2))
        
        x_axis = np.arange(0, w)
        gaussian_signal = np.exp(-0.5 * ((x_axis - target['smooth_x']) / dynamic_sigma) ** 2)
        alpha_map = np.maximum(alpha_map, gaussian_signal * dynamic_alpha)

    alpha_3d = np.dstack([alpha_map] * 3)
    overlay_green = np.zeros_like(frame)
    overlay_green[:, :] = [0, 255, 0]
    blended = (overlay_green * alpha_3d + frame * (1 - alpha_3d)).astype(np.uint8)

    # --- 10. HUD OVERLAY ---
    # Center Targeting line & Floating IMSI per active target
    for imsi, target in active_targets.items():
        # Skip if IMSI contains "RNTI" (case insensitive)
        if "RNTI" in imsi.upper():
            continue
            
        sx = target['smooth_x']
        color = (0, 255, 0)
        if target['smooth_mag'] < 5 or target.get('low_confidence'): color = (0, 150, 255) # Warning Amber if weak signal or unreliable sample
        
        # Vertical target line
        cv2.line(blended, (sx, 0), (sx, h), color, 2)
        
        # Floating IMSI near it
        imsi_text = f"IMSI: {imsi}"
        (tw, th), baseline = cv2.getTextSize(imsi_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Position floating text near the vertical line (at height h // 3)
        text_x = sx + 10
        text_y = h // 3
        
        # Keep inside frame boundaries
        if text_x + tw > w:
            text_x = sx - tw - 10
            
        cv2.rectangle(blended, (text_x - 5, text_y - th - 5), (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(blended, imsi_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw Compass
    draw_tactical_compass(blended, active_targets)
    
    # Telemetry Panel
    panel_h = 60 + len(active_targets) * 35
    if panel_h < 120: panel_h = 120
    
    cv2.rectangle(blended, (20, 20), (380, 20 + panel_h), (0, 0, 0), -1)
    cv2.putText(blended, "TACTICAL TARGETS", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add EARFCN and frequency info to telemetry panel
    freq_info = f"EARFCN: {EARFCN}  {FREQ_UL/1e6:.1f} MHz"
    cv2.putText(blended, freq_info, (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    
    y_offset = 110
    for imsi, target in active_targets.items():
        # Skip if IMSI contains "RNTI" (case insensitive)
        if "RNTI" in imsi.upper():
            continue
            
        text = f"{imsi[:15]}: {target['current_aoa']:+.1f} deg (MAG {target['current_mag']:.1f})"
        cv2.putText(blended, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 30
        
    cv2.putText(blended, f"PHASE CORR: {PHASE_CORRECTION:+.2f}  HFOV: {CAM_HFOV:.1f}", (40, 20 + panel_h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow(window_name, blended)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('a'): PHASE_CORRECTION -= 0.05
    elif key == ord('d'): PHASE_CORRECTION += 0.05
    elif key == ord('w'): CAM_HFOV += 1.0
    elif key == ord('s'): CAM_HFOV -= 1.0

cap.release()
cv2.destroyAllWindows()
