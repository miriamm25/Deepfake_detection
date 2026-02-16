#!/usr/bin/env python3
"""
03_manifest_distributions.py

Reads manifests/video_streams_manifest.csv and produces:
- counts over duration_bucket
- proportions of is_vfr
- proportions of has_audio == 0
- top resolutions
Plus simple plots and a summary.json

Usage:
  python3 scripts/03_manifest_distributions.py \
    --manifest manifests/video_streams_manifest.csv \
    --out_dir reports/manifest_stats
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = [
    "split",
    "status",
    "duration_bucket",
    "is_vfr",
    "has_audio",
    "width",
    "height",
    "duration_s",
]


def parse_args():
    """CLI arguments: where manifest is, and where outputs go."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to video_streams_manifest.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder for stats/plots")
    ap.add_argument("--status_filter", default="video_ok",
                    help="Which status to analyze (default: video_ok). Use 'all' to include everything.")
    return ap.parse_args()


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load CSV and validate columns exist."""
    df = pd.read_csv(manifest_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    return df


def filter_rows(df: pd.DataFrame, status_filter: str) -> pd.DataFrame:
    """
    Usually you want to compute distributions on usable videos only:
    status == video_ok.
    """
    if status_filter.lower() == "all":
        return df.copy()

    return df[df["status"] == status_filter].copy()


def add_resolution_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 'resolution' string like '576x1024'."""
    df["resolution"] = df["width"].astype("Int64").astype(str) + "x" + df["height"].astype("Int64").astype(str)
    return df


def duration_bucket_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per split and duration_bucket."""
    out = (
        df.groupby(["split", "duration_bucket"])
          .size()
          .reset_index(name="count")
          .sort_values(["split", "count"], ascending=[True, False])
    )
    return out


def vfr_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Share of is_vfr==1 per split."""
    g = df.groupby("split")["is_vfr"]
    out = pd.DataFrame({
        "n": g.size(),
        "vfr_count": g.sum(),
    }).reset_index()
    out["vfr_ratio"] = (out["vfr_count"] / out["n"]).round(4)
    return out


def audio_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Share of has_audio==0 per split."""
    g = df.groupby("split")["has_audio"]
    out = pd.DataFrame({
        "n": g.size(),
        "no_audio_count": (g.apply(lambda s: (s == 0).sum())),
    }).reset_index()
    out["no_audio_ratio"] = (out["no_audio_count"] / out["n"]).round(4)
    return out


def top_resolutions(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Top-K resolutions overall and per split."""
    overall = df["resolution"].value_counts().head(k).reset_index()
    overall.columns = ["resolution", "count"]
    return overall


def plot_duration_buckets(counts_df: pd.DataFrame, out_dir: Path):
    """Bar plots for duration_bucket distribution per split."""
    for split_name, sub in counts_df.groupby("split"):
        sub = sub.sort_values("count", ascending=False)
        plt.figure()
        plt.bar(sub["duration_bucket"].astype(str), sub["count"])
        plt.title(f"Duration bucket counts ({split_name})")
        plt.xlabel("duration_bucket")
        plt.ylabel("count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"duration_bucket_counts_{split_name}.png", dpi=150)
        plt.close()


def plot_top_resolutions(res_df: pd.DataFrame, out_dir: Path):
    """Bar plot for top resolutions (overall)."""
    plt.figure()
    plt.bar(res_df["resolution"], res_df["count"])
    plt.title("Top resolutions (overall)")
    plt.xlabel("resolution")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "top_resolutions_overall.png", dpi=150)
    plt.close()


def write_summary(df: pd.DataFrame, vfr_df: pd.DataFrame, audio_df: pd.DataFrame, out_dir: Path):
    """Write a small JSON with key metrics you’ll reference in decisions."""
    summary = {
        "rows_analyzed": int(len(df)),
        "splits": sorted(df["split"].dropna().unique().tolist()),
        "duration_s": {
            "min": float(df["duration_s"].min()),
            "median": float(df["duration_s"].median()),
            "p90": float(df["duration_s"].quantile(0.90)),
            "max": float(df["duration_s"].max()),
        },
        "vfr_by_split": vfr_df.to_dict(orient="records"),
        "no_audio_by_split": audio_df.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(manifest_path)
    df = filter_rows(df, args.status_filter)
    df = add_resolution_column(df)

    # Compute tables
    dur_counts = duration_bucket_counts(df)
    vfr_df = vfr_stats(df)
    aud_df = audio_stats(df)
    res_df = top_resolutions(df)

    # Save CSVs
    dur_counts.to_csv(out_dir / "duration_bucket_counts.csv", index=False)
    vfr_df.to_csv(out_dir / "vfr_by_split.csv", index=False)
    aud_df.to_csv(out_dir / "audio_presence_by_split.csv", index=False)
    res_df.to_csv(out_dir / "top_resolutions.csv", index=False)

    # Plots
    plot_duration_buckets(dur_counts, out_dir)
    plot_top_resolutions(res_df, out_dir)

    # Summary
    write_summary(df, vfr_df, aud_df, out_dir)

    print(f"[OK] Wrote stats to: {out_dir}")
    print(f"     Rows analyzed: {len(df)} (status_filter={args.status_filter})")


if __name__ == "__main__":
    main()
