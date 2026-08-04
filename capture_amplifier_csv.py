#!/usr/bin/env python3
"""
Converts the live DF UDP stream (chest_ul.c, "IMSI,spatial_delta,magnitude"
per packet on UDP:5555) into the CSV schema analyze_amp.py expects:

    frame, mean_phase, phase_var, mean_mag, mag_var, drift_rate_rps

This is the missing link mentioned during the code audit: chest_ul.c and the
DF bridge never produced this schema, so analyze_amp.py had no real data
source. Nothing in chest_ul.c needed to change for this — spatial_delta and
magnitude are already on the wire; this just buckets the raw per-TTI samples
into fixed-duration "frames" and computes per-frame stats over each bucket.

WORKFLOW
--------
Run this once WITHOUT the amplifier in the signal chain, then once WITH it,
pointing --out at the two filenames analyze_amp.py looks for by default:

    python3 capture_amplifier_csv.py --imsi <IMSI> --duration 60 \\
        --out /tmp/amplifier_analysis_wo_amplifier.csv

    # (attach the amplifier)
    python3 capture_amplifier_csv.py --imsi <IMSI> --duration 60 \\
        --out /tmp/amplifier_analysis.csv

    python3 analyze_amp.py   # reads both default paths automatically

Keep the reference UE/setup identical between the two runs (same position,
same distance) — the whole point is that DF/position noise stays constant so
whatever changes between the two files is attributable to the amplifier.

FRAME / STAT DEFINITIONS
-------------------------
- One "frame" = one --frame-ms window (default 200ms) of raw UDP packets.
- mean_phase / phase_var: circular mean and circular variance (1 - R, so it's
  bounded in [0, 1] and never blows up like -2*ln(R) does as R -> 0) of
  spatial_delta samples within the frame. Circular, not arithmetic, because
  spatial_delta wraps at +/-pi.
- mean_mag / mag_var: plain mean/variance of the magnitude samples in the frame.
- drift_rate_rps: (this frame's mean_phase - previous frame's mean_phase),
  wrapped to the shortest path and divided by the frame duration in seconds.
  First frame has no predecessor, so it's written as 0.0.
"""

import argparse
import csv
import math
import socket
import statistics
import sys
import time
from collections import defaultdict


def circular_stats(angles_rad):
    n = len(angles_rad)
    sin_sum = sum(math.sin(a) for a in angles_rad)
    cos_sum = sum(math.cos(a) for a in angles_rad)
    mean_angle = math.atan2(sin_sum / n, cos_sum / n)
    r = math.hypot(sin_sum / n, cos_sum / n)
    circ_var = 1.0 - r  # 0 = perfectly concentrated, 1 = uniformly spread
    return mean_angle, circ_var


def wrapped_diff(a, b):
    """Shortest signed distance from b to a, wrapped to [-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def main():
    ap = argparse.ArgumentParser(description="Bucket the DF UDP stream into analyze_amp.py's CSV schema")
    ap.add_argument("--port", type=int, default=5555, help="UDP bridge port (default: 5555)")
    ap.add_argument("--duration", type=float, default=60.0, help="Total capture time in seconds (default: 60)")
    ap.add_argument("--frame-ms", type=float, default=200.0, help="Frame bucket size in ms (default: 200)")
    ap.add_argument("--imsi", type=str, default=None, help="Only use this IMSI (default: auto-pick busiest)")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path, e.g. /tmp/amplifier_analysis.csv")
    args = ap.parse_args()

    frame_s = args.frame_ms / 1000.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.2)

    print("=" * 70)
    print(" AMPLIFIER CSV CAPTURE ".center(70, "="))
    print("=" * 70)
    print(f"\nListening on UDP:{args.port} for {args.duration:.0f}s, {args.frame_ms:.0f}ms frames -> {args.out}")
    if args.imsi:
        print(f"Filtering to IMSI={args.imsi}")
    else:
        print("No --imsi given: will lock onto whichever IMSI sends the first packet.")
    print("(Ctrl+C to stop early; whatever was captured so far is kept)\n")

    target_imsi = args.imsi
    counts_seen = defaultdict(int)

    frame_idx = 0
    prev_mean_phase = None
    bucket_deltas = []
    bucket_mags = []
    bucket_start = None
    rows_written = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "mean_phase", "phase_var", "mean_mag", "mag_var", "drift_rate_rps"])
        f.flush()

        run_start = time.time()
        try:
            while time.time() - run_start < args.duration:
                try:
                    data, _ = sock.recvfrom(256)
                except socket.timeout:
                    now = time.time()
                    if bucket_start is not None and now - bucket_start >= frame_s and bucket_deltas:
                        frame_idx, prev_mean_phase, rows_written = _flush_frame(
                            writer, f, frame_idx, bucket_deltas, bucket_mags, prev_mean_phase, frame_s, rows_written
                        )
                        bucket_deltas, bucket_mags, bucket_start = [], [], None
                    continue

                try:
                    # Wire format is "imsi,delta,mag,csi_var"; accept the older
                    # 3-field format too. csi_var isn't used by this schema.
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

                counts_seen[imsi] += 1
                if target_imsi is None:
                    target_imsi = imsi
                    print(f"Locked onto IMSI={target_imsi}\n")
                if imsi != target_imsi:
                    continue

                now = time.time()
                if bucket_start is None:
                    bucket_start = now
                bucket_deltas.append(delta)
                bucket_mags.append(mag)

                if now - bucket_start >= frame_s:
                    frame_idx, prev_mean_phase, rows_written = _flush_frame(
                        writer, f, frame_idx, bucket_deltas, bucket_mags, prev_mean_phase, frame_s, rows_written
                    )
                    bucket_deltas, bucket_mags, bucket_start = [], [], None
                    print(f"  frames written: {rows_written}", end="\r")
        except KeyboardInterrupt:
            print("\n\nStopped early by user.")

        # Flush a final partial frame so short runs still produce something.
        if bucket_deltas:
            frame_idx, prev_mean_phase, rows_written = _flush_frame(
                writer, f, frame_idx, bucket_deltas, bucket_mags, prev_mean_phase, frame_s, rows_written
            )

    print(f"\n\nWrote {rows_written} frames to {args.out}")
    if target_imsi is None:
        print("\nWARNING: never saw any UDP packets at all. Check srsenb is running with")
        print("the patched chest_ul.c and that the reference UE is actively transmitting.")
        sys.exit(1)
    if rows_written == 0:
        print(f"\nWARNING: 0 frames written for IMSI={target_imsi}. Nothing for analyze_amp.py to read.")
        sys.exit(1)
    other = {k: v for k, v in counts_seen.items() if k != target_imsi}
    if other:
        print(f"NOTE: also saw traffic from other IMSIs (ignored): {dict(other)}")


def _flush_frame(writer, f, frame_idx, deltas, mags, prev_mean_phase, frame_s, rows_written):
    mean_phase, phase_var = circular_stats(deltas)
    mean_mag = statistics.fmean(mags)
    mag_var = statistics.variance(mags) if len(mags) > 1 else 0.0

    if prev_mean_phase is None:
        drift_rate_rps = 0.0
    else:
        drift_rate_rps = wrapped_diff(mean_phase, prev_mean_phase) / frame_s

    writer.writerow([frame_idx, f"{mean_phase:.6f}", f"{phase_var:.6f}", f"{mean_mag:.4f}", f"{mag_var:.4f}", f"{drift_rate_rps:.6f}"])
    f.flush()
    return frame_idx + 1, mean_phase, rows_written + 1


if __name__ == "__main__":
    main()
