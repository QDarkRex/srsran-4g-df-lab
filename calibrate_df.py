#!/usr/bin/env python3
"""
DF phase calibration tool for the 2-antenna spatial_delta pipeline in chest_ul.c.

WHAT THIS MEASURES
-------------------
Any static phase offset between the RX0/RX1 signal chains (cable length,
connector, LNA phase response) shifts every spatial_delta reading by a
constant amount. Until that offset is measured and removed, "CENTER" in the
console dashboard does not correspond to true broadside, and the LEFT/RIGHT
threshold (+/-0.20 rad) is biased.

This script does not care *how* you generate the reference signal — it just
listens on the same UDP:5555 stream the eNB already broadcasts
("IMSI,spatial_delta,magnitude" per packet) and averages spatial_delta while
that reference is active. Two ways to produce the reference, pick whichever
you can do first:

  1. UE broadside: put one test UE (SIM known) directly in front of / along
     the perpendicular bisector of the RX0/RX1 antenna pair, at a fixed
     distance, and keep it transmitting (e.g. iperf/ping) for the capture
     window. Least precise (multipath + real air interface adds noise) but
     needs no extra hardware.
  2. RF loopback: split one transmit source into both RX0 and RX1 with a
     power splitter/combiner. The measured offset is then purely the
     cable/connector/hardware phase difference, with no air interface noise.
     More precise; use this once you have a splitter on hand.

Either way, run this script *while the reference is transmitting*, and only
that IMSI should be active (see --imsi to filter if other UEs are on the
cell).

USAGE
-----
    python3 calibrate_df.py --imsi 001010000000001 --duration 30
    python3 calibrate_df.py --duration 30            # auto-picks the busiest IMSI

After it finishes it writes a single float (radians) to /tmp/df_calibration.conf.
chest_ul.c reads that file ONCE at process start and subtracts it from every
spatial_delta before it's used for classification/export — restart srsenb
for a new calibration to take effect.

To recalibrate later: delete /tmp/df_calibration.conf (or leave the old one,
this script always reports the *new* offset — see NOTE in the report about
the old offset already being subtracted upstream).
"""

import argparse
import math
import socket
import sys
import time
from collections import defaultdict


def circular_mean_std(angles_rad):
    """Proper circular statistics: a plain arithmetic mean is wrong for
    angles that can sit near the +/-pi wraparound point."""
    sin_sum = sum(math.sin(a) for a in angles_rad)
    cos_sum = sum(math.cos(a) for a in angles_rad)
    n = len(angles_rad)
    mean_angle = math.atan2(sin_sum / n, cos_sum / n)
    r = math.hypot(sin_sum / n, cos_sum / n)  # resultant length, 0..1 (1 = no spread)
    circ_std = math.sqrt(-2.0 * math.log(r)) if r > 1e-9 else float("inf")
    return mean_angle, circ_std, r


def main():
    ap = argparse.ArgumentParser(description="Calibrate static RX0/RX1 phase offset for DF spatial_delta")
    ap.add_argument("--port", type=int, default=5555, help="UDP bridge port (default: 5555, matches chest_ul.c)")
    ap.add_argument("--duration", type=float, default=30.0, help="Capture window in seconds (default: 30)")
    ap.add_argument("--imsi", type=str, default=None, help="Only use samples from this IMSI (default: auto-pick busiest)")
    ap.add_argument("--min-samples", type=int, default=100, help="Minimum samples required before trusting the result")
    ap.add_argument("--out", type=str, default="/tmp/df_calibration.conf", help="Where to write the offset (radians)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    print("=" * 70)
    print(" DF PHASE CALIBRATION ".center(70, "="))
    print("=" * 70)
    print(f"\nListening on UDP:{args.port} for {args.duration:.0f}s ...")
    if args.imsi:
        print(f"Filtering to IMSI={args.imsi}")
    else:
        print("No --imsi given: will auto-pick whichever IMSI sends the most packets.")
    print("(Ctrl+C to stop early and use whatever was captured so far)\n")

    samples_by_imsi = defaultdict(list)  # imsi -> list of (delta_rad, mag)
    start = time.time()
    last_report = start
    total_pkts = 0

    try:
        while time.time() - start < args.duration:
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            try:
                # Wire format is "imsi,delta,mag,csi_var" (csi_var added so
                # consumers can gate on signal quality instead of trusting
                # every raw sample). Also accept the older 3-field format.
                parts = data.decode(errors="ignore").strip().split(",")
                if len(parts) == 4:
                    imsi, delta_str, mag_str, _csi_var_str = parts
                elif len(parts) == 3:
                    imsi, delta_str, mag_str = parts
                else:
                    continue
                delta = float(delta_str)
                mag = float(mag_str)
            except ValueError:
                continue

            total_pkts += 1
            if args.imsi is None or imsi == args.imsi:
                samples_by_imsi[imsi].append((delta, mag))

            now = time.time()
            if now - last_report >= 1.0:
                elapsed = now - start
                counts = {k: len(v) for k, v in samples_by_imsi.items()}
                print(f"  [{elapsed:5.1f}s] total_pkts={total_pkts} samples_per_imsi={counts}", end="\r")
                last_report = now
    except KeyboardInterrupt:
        print("\n\nStopped early by user.")

    print("\n")

    if not samples_by_imsi:
        print("No UDP packets received at all. Check that:")
        print("  - srsenb is running with the patched chest_ul.c")
        print("  - the reference UE/UL signal is actually active during the capture window")
        print("  - nothing else is bound to UDP:5555 already")
        sys.exit(1)

    if args.imsi:
        target_imsi = args.imsi
        if target_imsi not in samples_by_imsi:
            print(f"Never saw any packets for IMSI={target_imsi}. Saw instead: {list(samples_by_imsi.keys())}")
            sys.exit(1)
    else:
        target_imsi = max(samples_by_imsi, key=lambda k: len(samples_by_imsi[k]))
        other_imsis = [k for k in samples_by_imsi if k != target_imsi]
        if other_imsis:
            print(f"NOTE: also saw traffic from {other_imsis} — ignoring those, using only '{target_imsi}'.")
            print("      If that's not your reference UE, rerun with --imsi to be explicit.\n")

    deltas = [d for d, m in samples_by_imsi[target_imsi]]
    mags = [m for d, m in samples_by_imsi[target_imsi]]
    n = len(deltas)

    print(f"Target IMSI: {target_imsi}")
    print(f"Samples:     {n}")

    if n < args.min_samples:
        print(f"\n  INSUFFICIENT DATA: got {n} samples, need at least {args.min_samples}.")
        print("  Not writing a calibration file. Let the reference transmit longer / more")
        print("  often (e.g. continuous ping or iperf) and rerun.")
        sys.exit(1)

    mean_angle, circ_std, r = circular_mean_std(deltas)
    mag_sorted = sorted(mags)
    mag_median = mag_sorted[n // 2]

    print(f"\nRaw spatial_delta over capture window:")
    print(f"  circular mean (= measured offset): {mean_angle:+.4f} rad ({math.degrees(mean_angle):+.2f} deg)")
    print(f"  circular std dev:                  {circ_std:.4f} rad")
    print(f"  resultant length R (1.0 = perfect): {r:.4f}")
    print(f"  magnitude (median):                {mag_median:.2f}")

    print("\n" + "-" * 70)
    if r < 0.85:
        print("WARNING: R < 0.85 — spatial_delta is spread out a lot during the capture.")
        print("  Likely causes: reference UE not actually stationary/broadside, weak/")
        print("  multipath-heavy signal, or something else moving in the environment.")
        print("  You can still write this offset, but treat it as a rough first pass and")
        print("  re-run once the setup is more controlled (loopback splitter removes most")
        print("  of this uncertainty since there's no air interface at all).")
    else:
        print("R looks reasonably tight — offset should be usable as-is.")
    print("-" * 70)

    with open(args.out, "w") as f:
        f.write(f"{mean_angle:.6f}\n")

    print(f"\nWrote offset to {args.out}: {mean_angle:+.6f} rad")
    print("\nNEXT STEPS:")
    print("  1. Restart srsenb (the offset is only loaded once, at process start).")
    print("  2. Re-check the console dashboard with the reference UE back in the SAME")
    print("     broadside position — it should now read close to CENTER (~0.00).")
    print("  3. Move the reference to a known LEFT/RIGHT position and confirm the")
    print("     dashboard direction matches reality before trusting live readings.")


if __name__ == "__main__":
    main()
