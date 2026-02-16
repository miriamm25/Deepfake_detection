#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return r.fieldnames or [], rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def stem_no_ext(filename: str) -> str:
    # "abc.mp4" -> "abc"
    name = (filename or "").strip()
    if name.lower().endswith(".mp4"):
        return name[:-4]
    return Path(name).stem if name else ""


def normalize_vgt(x: str) -> str:
    x = (x or "").strip().lower()
    if x == "real":
        return "real"
    if x == "fake":
        return "fake"
    if x in {"unknown", ""}:
        return "unknown"
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repo root (.)")
    ap.add_argument("--clips_std_manifest", default="manifests/clips_std_manifest.csv")
    ap.add_argument("--video_metadata", default="video_metadata_enhanced.csv")
    ap.add_argument("--out_manifest", default="manifests/clips_std_labeled_manifest.csv")
    ap.add_argument("--report_dir", default="reports/clips_std_labeled")
    ap.add_argument("--keep_unknown", action="store_true",
                    help="If set, keep clips whose video label is unknown; otherwise drop them.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    clips_m = root / args.clips_std_manifest
    meta_p = root / args.video_metadata
    out_m = root / args.out_manifest
    report_dir = root / args.report_dir
    ensure_dir(report_dir)

    if not clips_m.exists():
        raise FileNotFoundError(f"Missing clips_std manifest: {clips_m}")
    if not meta_p.exists():
        raise FileNotFoundError(f"Missing video metadata: {meta_p}")

    _, clips_rows = read_csv(clips_m)
    _, meta_rows = read_csv(meta_p)

    # Build mapping: video_id(stem) -> (label, split_from_metadata optional)
    vid2 = {}
    split_meta = {}
    for r in meta_rows:
        vid = stem_no_ext(r.get("Filename", ""))
        if not vid:
            continue

        # Prefer explicit GT column "Video Ground Truth"
        vgt = normalize_vgt(r.get("Video Ground Truth", ""))

        # If you want to be ultra-safe, you can derive from booleans:
        # is_video_fake/is_video_real
        is_fake = (r.get("is_video_fake", "") or "").strip().lower() == "true"
        is_real = (r.get("is_video_real", "") or "").strip().lower() == "true"
        if is_fake and not is_real:
            label = "fake"
        elif is_real and not is_fake:
            label = "real"
        else:
            label = vgt if vgt in {"real", "fake"} else "unknown"

        vid2[vid] = label
        split_meta[vid] = (r.get("split") or "").strip().lower()

    out_rows = []
    counters = Counter()
    missing = set()
    unknown = set()
    skipped_non_ok = 0
    skipped_ffmpeg_fail = 0

    for r in clips_rows:
        # IMPORTANT: clips_std_manifest are "output rows" from Script 06.
        # Keep only successful encodes:
        if str(r.get("ffmpeg_rc", "")).strip() != "0":
            skipped_ffmpeg_fail += 1
            continue

        split = (r.get("split") or "").strip().lower()
        video_id = (r.get("video_id") or "").strip()
        clip_id = (r.get("clip_id") or "").strip()

        label = vid2.get(video_id, "")
        if not label:
            missing.add(video_id)
            label = "unknown"

        if label == "unknown":
            unknown.add(video_id)
            if not args.keep_unknown:
                skipped_non_ok += 1
                continue

        # Optional consistency check: manifest split vs metadata split
        meta_split = split_meta.get(video_id, "")
        split_mismatch = 1 if (meta_split and split and meta_split != split) else 0

        out = dict(r)
        out["label"] = label
        out["split_mismatch_vs_metadata"] = str(split_mismatch)
        out_rows.append(out)

        counters[(split, label)] += 1

    # Write
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    if "label" not in fieldnames:
        fieldnames.append("label")
    if "split_mismatch_vs_metadata" not in fieldnames:
        fieldnames.append("split_mismatch_vs_metadata")
    write_csv(out_m, fieldnames, out_rows)

    (report_dir / "missing_video_ids.txt").write_text("\n".join(sorted(missing)), encoding="utf-8")
    (report_dir / "unknown_video_ids.txt").write_text("\n".join(sorted(unknown)), encoding="utf-8")

    summary = {
        "rows_in_clips_std_manifest": len(clips_rows),
        "rows_out_labeled": len(out_rows),
        "skipped_ffmpeg_fail": skipped_ffmpeg_fail,
        "skipped_unknown_label": skipped_non_ok if not args.keep_unknown else 0,
        "missing_video_ids_in_metadata": len(missing),
        "unknown_video_ids": len(unknown),
        "counts_by_split_label": {f"{k[0]}::{k[1]}": v for k, v in counters.items()},
        "out_manifest": str(out_m.relative_to(root)),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print("Output:", out_m)
    print("Report:", report_dir / "summary.json")
    print("Counts:", summary["counts_by_split_label"])


if __name__ == "__main__":
    main()
