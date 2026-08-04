# DF + Amplifier Testing Checklist

Status: software side audited and patched (race condition, hot-path I/O, single-subcarrier
phase, UDP destination bug, calibration, leaked Telegram token, dual-calibration risk,
camera-jitter-with-amplifier hardening — see git log). Second pass: mutex restructured so
file/network I/O never holds the PHY worker lock; sprintf→snprintf; socket error checks;
if(true) dead branch removed; EARFCN auto-validation vs enb.conf; Telegram polling backoff;
pvariance→variance for small sample frames; __pycache__ cleaned from repo. This checklist
is for the first real hardware run on the lab machine.

## 0. Known symptom already addressed in software: camera jitter when amplifier is attached

The programmer reported the camera HUD gets jittery once the amplifier is in the chain. Two
software-side contributors were found and fixed:

1. `chest_ul.c`'s DF UDP packet now also carries the multipath/quality indicator
   (`csi_var`, 4th field). `radio_ar_desktop.py` uses it to freeze the plotted angle
   instead of drawing a noisy/impossible reading (also catches the case where phase noise
   pushes the AoA math to a physically invalid angle, which used to snap the marker to the
   frame edge). Low-confidence samples now render amber instead of green.
2. `analyze_amp.py` gained a clipping/saturation heuristic: if magnitude variance didn't
   grow the way it should for the observed gain while phase noise did, it now prints an
   explicit "POSSIBLE RX CLIPPING/SATURATION" warning.

**This is very likely a `rx_gain` problem, not a bug in the amplifier or the DF math**:
`srsenb/enb.conf`'s `rx_gain = 80` is a fixed value that was almost certainly tuned
*without* the amplifier attached. With the amplifier now boosting the input, that same
`rx_gain` can push the B210's ADC into saturation/clipping — which corrupts the channel
estimate and shows up as exactly this kind of jitter. **Try lowering `rx_gain` while the
amplifier is attached and re-run Test 5 below before assuming the amplifier is faulty.**

## 1. What to copy to the Linux lab machine

**Required:**
- This whole repo (`srsran-4g-df1/`) — includes all patches: `chest_ul.c`, `calibrate_df.py`,
  `capture_amplifier_csv.py`, `analyze_amp.py`, `scripts/radio_ar_desktop.py`, `enb.conf`,
  `user_db.csv`.

**Optional (only if needed):**
- `LibreSDR_USRP/` — only if flashing LibreSDR firmware onto the B210.
- `mediamtx/` — only if the camera is a networked IP camera, not a local USB webcam
  (`radio_ar_desktop.py` uses `cv2.VideoCapture(0)` — a local webcam — by default).

**Not needed for this test:** `sigover/`, `plane/`.

## 2. Install dependencies on the lab machine

```bash
sudo apt update
sudo apt install -y cmake build-essential libfftw3-dev libmbedtls-dev libboost-program-options-dev libconfig++-dev libsctp-dev libczmq-dev uhd-host libuhd-dev python3-pip
sudo uhd_images_downloader
pip3 install -r requirements.txt
```

## 3. Build

```bash
cd srsran-4g-df1
mkdir build && cd build
cmake ../
make -j$(nproc)
```

Should succeed with 0 errors, same as the WSL validation build. If something differs here,
that's a lab-specific dependency mismatch — send the build log back.

## 4. Physical checklist before powering anything on

- [ ] B210 on USB **3.0** (not 2.0)
- [ ] RX0 and RX1 cables are the **same length**
- [ ] `uhd_find_devices` detects the B210
- [ ] At least one test SIM registered in `srsepc/user_db.csv`
- [ ] Amplifier **not** attached yet for the first run (establish a baseline first)
- [ ] USB webcam connected if testing the camera HUD

## 5. Test sequence — do not skip ahead

> **Important:** `radio_ar_desktop.py`, `calibrate_df.py`, and `capture_amplifier_csv.py` all
> bind to the same UDP port (5555). Only one of them can run at a time — whichever binds
> first gets the data, the others get nothing.

### Test 1 — Basic bring-up (no DF/camera tooling yet)

```bash
cd srsepc/src && sudo ./srsepc epc.conf
cd srsenb/src && sudo ./srsenb enb.conf
```

- **Pass:** UE attaches, console dashboard shows the **real IMSI** (not `RNTI-0x...`
  indefinitely). If IMSI never resolves, check `/tmp/rnti_imsi.csv` etc. actually have data.
- **Save:** full `srsenb` console log (not just the dashboard).

### Test 2 — Phase calibration (the only UDP:5555 consumer running at this point)

Position one UE perpendicular to the antenna line, fixed distance, keep it transmitting
(ping/iperf):

```bash
python3 calibrate_df.py --imsi <UE_IMSI> --duration 30
```

- **Pass:** resultant length R ≥ 0.85, sample count ≥ 100.
- Then **restart `srsenb`** — the offset is only read once at process start.
- **Save:** the resulting `/tmp/df_calibration.conf` + a screenshot of the calibration output.

### Test 3 — Direction validation (console dashboard first, camera HUD after)

Move the UE to left/right positions you know the exact angle of. Compare against the
console dashboard.

- **Pass:** broadside (0°) → CENTER, left → LEFT, right → RIGHT, consistent across
  several positions.
- If this fails, **do not proceed to amplifier/camera testing** — this is the prerequisite
  for everything else. Usually means calibration wasn't done right, or RX0/RX1 cables
  aren't matched length.

### Test 4 — Camera HUD (optional; stop `calibrate_df.py` first — port conflict)

```bash
python3 scripts/radio_ar_desktop.py --target-imsi <UE_IMSI>
```

Before running: the script now auto-validates `EARFCN` against `srsenb/enb.conf` at
startup and prints a warning (and auto-corrects) if they mismatch. Leave
`PHASE_CORRECTION` at `0.0` — the offset is already applied upstream from Test 2.

- **Pass:** the target line on video follows the UE's real position as it moves left/right.
- **Save:** recording/screenshots of the HUD at several known UE positions.

### Test 5 — Amplifier (stop other UDP tools first)

```bash
python3 capture_amplifier_csv.py --imsi <UE_IMSI> --duration 60 --out /tmp/amplifier_analysis_wo_amplifier.csv
# attach the amplifier now — do NOT move the UE or change its distance
python3 capture_amplifier_csv.py --imsi <UE_IMSI> --duration 60 --out /tmp/amplifier_analysis.csv
python3 analyze_amp.py
```

- **Read:** gain in dB, phase noise increase, surge %.
- **Save:** both CSVs + `/tmp/amp_comparison.png` + the full verdict text (not just the
  final conclusion line).

## 6. If something looks wrong, bring back:

1. Full `srsenb` console log from that run.
2. Files from `/tmp/`: `csi_capture.bin`, `multipath_<IMSI>.csv`, `df_calibration.conf`,
   both amplifier CSVs.
3. Which test number failed + what was expected vs. what happened.
4. Physical UE/antenna position at the time (photo or noted angle).

With that, a bug vs. a calibration mistake vs. a real hardware limit can be told apart
without guessing.

## Known non-code caveats (not fixable in software)

- `srsenb/enb.conf`'s `tx_gain` is currently `89` (was `69` in the original upstream-tracked
  version) — confirm this jump was intentional before transmitting at full power in a
  non-shielded room.
- RX0/RX1 cable length matching and antenna spacing (≤ λ/2 for the LTE band in use) are
  physical prerequisites for Test 2/3 that no amount of software can substitute for.
