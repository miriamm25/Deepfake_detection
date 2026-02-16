#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple

import pandas as pd


# -----------------------------
# Helpers: derived grouping
# -----------------------------
def derive_orientation(w: float, h: float) -> str:
    if pd.isna(w) or pd.isna(h) or w <= 0 or h <= 0:
        return "unknown"
    if abs(w - h) <= 1:
        return "square"
    return "vertical" if h > w else "horizontal"


def derive_fps_bucket(fps: float, is_vfr: float) -> str:
    # is_vfr in your manifest is 0/1 (but keep robust)
    if str(is_vfr).strip() in {"1", "True", "true"}:
        return "vfr"
    if pd.isna(fps) or fps <= 0:
        return "unknown"
    if fps <= 25:
        return "<=25"
    if 28 <= fps <= 32:
        return "~30"
    if fps >= 50:
        return ">=50"
    return "other"


def derive_audio_group(has_audio: float) -> str:
    if str(has_audio).strip() in {"1", "True", "true"}:
        return "has_audio"
    if str(has_audio).strip() in {"0", "False", "false"}:
        return "no_audio"
    return "unknown"


def uniform_windows(duration_s: float, clip_len_s: float, n: int) -> List[Tuple[float, float]]:
    """
    Returns n windows [start, end] inside [0, duration_s].
    If duration < clip_len, returns [0, duration] as single window (n will be ignored upstream).
    For duration >= clip_len: windows are roughly uniformly spaced.
    """
    if duration_s <= 0:
        return []
    if duration_s <= clip_len_s:
        return [(0.0, float(duration_s))]

    max_start = duration_s - clip_len_s
    if n <= 1:
        return [(0.0, clip_len_s)]

    # Uniformly spaced starts from 0..max_start
    starts = [i * (max_start / (n - 1)) for i in range(n)]
    return [(float(s), float(s + clip_len_s)) for s in starts]


def plan_clips_for_row(row: pd.Series, clip_len_s: float, max_clips: int) -> List[dict]:
    """
    Dense non-overlapping clip extraction with a per-video cap.
    n_clips = min(floor(duration / clip_len), max_clips).
    """
    vid = row["video_id"]
    split = row["split"]
    duration_s = float(row.get("duration_s", 0) or 0)
    bucket = row.get("duration_bucket", "unknown")

    raw_n = max(1, int(duration_s // clip_len_s))
    n_clips = min(raw_n, max_clips)
    capped = raw_n > max_clips
    strategy = "dense_capped" if capped else "dense"

    windows = uniform_windows(duration_s, clip_len_s, n_clips)

    out = []
    for i, (s, e) in enumerate(windows):
        clip_id = f"{vid}__c{i:02d}"
        out.append(
            {
                "video_id": vid,
                "split": split,
                "clip_id": clip_id,
                "start_s": round(s, 3),
                "end_s": round(e, 3),
                "clip_len_s": round(e - s, 3),
                "duration_bucket": bucket,
                "strategy_tag": strategy,
                "orientation": row.get("orientation", "unknown"),
                "fps_bucket": row.get("fps_bucket", "unknown"),
                "audio_group": row.get("audio_group", "unknown"),
            }
        )
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifests/video_streams_manifest.csv")
    ap.add_argument("--out_manifest", default="manifests/clips_manifest.csv")
    ap.add_argument("--out_dir", default="reports/clip_plan")
    ap.add_argument("--status_filter", default="video_ok", help="Only plan clips for this status (default: video_ok)")
    ap.add_argument("--clip_len", type=float, default=5.0, help="Clip length in seconds")
    ap.add_argument("--max_clips_per_video", type=int, default=24, help="Cap clips per video (24 = 2min coverage)")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)

    # Filter
    if "status" in df.columns:
        df = df[df["status"] == args.status_filter].copy()

    # Derived grouping columns (do it once here so all later scripts can reuse)
    df["orientation"] = df.apply(lambda r: derive_orientation(r.get("width"), r.get("height")), axis=1)
    df["fps_bucket"] = df.apply(lambda r: derive_fps_bucket(r.get("fps"), r.get("is_vfr")), axis=1)
    df["audio_group"] = df.apply(lambda r: derive_audio_group(r.get("has_audio")), axis=1)

    planned = []
    for _, row in df.iterrows():
        planned.extend(plan_clips_for_row(row, args.clip_len, args.max_clips_per_video))

    clips_df = pd.DataFrame(planned)

    # Make output dirs
    os.makedirs(os.path.dirname(args.out_manifest) or ".", exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    clips_df.to_csv(args.out_manifest, index=False)

    # Summary KPI
    summary = {
        "input_manifest": args.manifest,
        "rows_video_ok": int(len(df)),
        "clips_total": int(len(clips_df)),
        "clips_by_split": clips_df.groupby("split")["clip_id"].count().to_dict(),
        "clips_by_bucket": clips_df.groupby("duration_bucket")["clip_id"].count().to_dict(),
        "clips_by_orientation": clips_df.groupby("orientation")["clip_id"].count().to_dict(),
        "clips_by_fps_bucket": clips_df.groupby("fps_bucket")["clip_id"].count().to_dict(),
        "clips_by_audio_group": clips_df.groupby("audio_group")["clip_id"].count().to_dict(),
        "clip_len_s": args.clip_len,
        "max_clips_per_video": args.max_clips_per_video,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote: {args.out_manifest}")
    print(f"[OK] Wrote: {os.path.join(args.out_dir, 'summary.json')}")
    print(f"[OK] clips_total={len(clips_df)}")


if __name__ == "__main__":
    main()
