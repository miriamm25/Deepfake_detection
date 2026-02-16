#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

def infer_split(p: Path) -> str:
    parts = p.as_posix().split("/")
    # expected: clips_std/{split}/{video_id}/{clip_id}.mp4
    if len(parts) >= 3 and parts[-4] == "clips_std":
        return parts[-3]
    # fallback: look for train/test in path
    for s in ("train", "test"):
        if f"/{s}/" in p.as_posix():
            return s
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repo root, e.g. .")
    ap.add_argument("--clips_std_dir", default="clips_std", help="Folder with standardized clips")
    ap.add_argument("--out", default="manifests/clips_std_manifest.csv", help="Output CSV path")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    clips_std = (root / args.clips_std_dir).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for mp4 in sorted(clips_std.rglob("*.mp4")):
        # mp4: clips_std/train/<video_id>/<clip_id>.mp4
        rel = mp4.relative_to(root).as_posix()
        parts = rel.split("/")
        split = infer_split(mp4)

        video_id = ""
        clip_id = mp4.stem
        if len(parts) >= 4 and parts[-4] == "clips_std":
            # clips_std/train/<video_id>/<clip>.mp4
            video_id = parts[-2]

        rows.append({
            "split": split,
            "video_id": video_id,
            "clip_id": clip_id,
            "std_rel_path": rel,
            "std_abs_path": mp4.as_posix(),
            "status": "ok",
            "error_msg": "",
        })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "split","video_id","clip_id","std_rel_path","std_abs_path","status","error_msg"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Done.")
    print(f"Found standardized clips: {len(rows)}")
    print(f"Output manifest: {out_path}")

if __name__ == "__main__":
    main()
