#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple

def run_ffmpeg_extract(ffmpeg: str, src: Path, dst: Path, start_s: float, end_s: float) -> Tuple[bool, str]:
    """
    Extrage un subclip [start_s, end_s] din src în dst folosind ffmpeg.

    - Folosim re-encode (libx264 + aac) pentru compatibilitate + stabilitate la training.
      (Copy-stream e mai rapid, dar poate da probleme la keyframes / timebase / seeking.)
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    duration = max(0.0, end_s - start_s)
    if duration <= 0.05:
        return False, f"Invalid duration computed: {duration:.4f}s"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_s:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        str(dst),
    ]

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return False, p.stderr.strip()[:1000]
        if not dst.exists() or dst.stat().st_size < 1024:
            return False, "Output file missing or too small."
        return True, ""
    except Exception as e:
        return False, f"Exception: {e}"

def safe_clip_id(existing: str, start_s: float, end_s: float) -> str:
    """
    Asigură un nume de clip stabil și informativ.
    Dacă ai deja clip_id în manifest, îl păstrăm, dar dacă e gol,
    generăm unul din start/end (în centisecunde).
    """
    if existing and existing.strip():
        return existing.strip()

    s_cs = int(round(start_s * 100))  # centiseconds
    e_cs = int(round(end_s * 100))
    return f"s{s_cs:08d}_e{e_cs:08d}"

def load_video_index(video_manifest_csv: Path) -> Dict[Tuple[str, str], str]:
    """
    Citește video_streams_manifest.csv și construiește un index:
    (split, video_id) -> rel_path
    ca să găsim rapid fișierul sursă.
    """
    idx = {}
    with video_manifest_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"split", "video_id", "rel_path", "status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"video_streams_manifest.csv missing columns: {missing}")

        for row in reader:
            if row.get("status") != "video_ok":
                continue
            split = row["split"]
            vid = row["video_id"]
            rel = row["rel_path"]
            idx[(split, vid)] = rel
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root-ul repo-ului Deepfake-Eval-2024 (folderul cu manifests/, organized/ etc.)")
    ap.add_argument("--clips_manifest", default="manifests/clips_manifest.csv", help="Planul de clipuri (relativ la --root).")
    ap.add_argument("--video_manifest", default="manifests/video_streams_manifest.csv", help="Manifestul video streams (relativ la --root).")
    ap.add_argument("--out_dir", default="clips", help="Folderul unde salvăm clipurile (relativ la --root).")
    ap.add_argument("--out_manifest", default="manifests/clips_on_disk_manifest.csv", help="Manifest output (relativ la --root).")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Calea către ffmpeg (implicit: ffmpeg din PATH).")
    ap.add_argument("--dry_run", action="store_true", help="Nu extrage nimic, doar raportează ce ar face.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    clips_manifest = (root / args.clips_manifest).resolve()
    video_manifest = (root / args.video_manifest).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_manifest = (root / args.out_manifest).resolve()

    if not clips_manifest.exists():
        raise FileNotFoundError(f"Missing clips_manifest: {clips_manifest}")
    if not video_manifest.exists():
        raise FileNotFoundError(f"Missing video_manifest: {video_manifest}")

    # Index ca să știm unde e video-ul sursă
    video_idx = load_video_index(video_manifest)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    # Scriem manifestul nou: fiecare clip + unde e pe disk + status
    out_fields = [
        "split", "video_id", "clip_id", "start_s", "end_s",
        "src_rel_path", "src_abs_path",
        "clip_rel_path", "clip_abs_path",
        "status", "error_msg"
    ]

    n_total = 0
    n_ok = 0
    n_fail = 0
    n_missing_src = 0

    with clips_manifest.open("r", newline="", encoding="utf-8") as f_in, \
         out_manifest.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        required = {"split", "video_id", "start_s", "end_s"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"clips_manifest.csv missing columns: {missing}")

        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            n_total += 1

            split = row["split"].strip()
            vid = row["video_id"].strip()
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            clip_id = safe_clip_id(row.get("clip_id", ""), start_s, end_s)

            src_rel = video_idx.get((split, vid), "")
            if not src_rel:
                n_missing_src += 1
                writer.writerow({
                    "split": split,
                    "video_id": vid,
                    "clip_id": clip_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "src_rel_path": "",
                    "src_abs_path": "",
                    "clip_rel_path": "",
                    "clip_abs_path": "",
                    "status": "fail_missing_source",
                    "error_msg": "Video not found in video_streams_manifest.csv (or not video_ok)."
                })
                n_fail += 1
                continue

            src_abs = (root / src_rel).resolve()
            if not src_abs.exists():
                n_missing_src += 1
                writer.writerow({
                    "split": split,
                    "video_id": vid,
                    "clip_id": clip_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "src_rel_path": src_rel,
                    "src_abs_path": str(src_abs),
                    "clip_rel_path": "",
                    "clip_abs_path": "",
                    "status": "fail_missing_file",
                    "error_msg": "Source video file does not exist on disk."
                })
                n_fail += 1
                continue

            # Unde scriem clipul
            clip_rel = Path(args.out_dir) / split / vid / f"{clip_id}.mp4"
            clip_abs = (root / clip_rel).resolve()

            if args.dry_run:
                ok, err = True, ""
            else:
                ok, err = run_ffmpeg_extract(args.ffmpeg, src_abs, clip_abs, start_s, end_s)

            status = "ok" if ok else "fail_extract"
            if ok:
                n_ok += 1
            else:
                n_fail += 1

            writer.writerow({
                "split": split,
                "video_id": vid,
                "clip_id": clip_id,
                "start_s": start_s,
                "end_s": end_s,
                "src_rel_path": src_rel,
                "src_abs_path": str(src_abs),
                "clip_rel_path": str(clip_rel),
                "clip_abs_path": str(clip_abs),
                "status": status,
                "error_msg": err
            })

    print("Done.")
    print(f"Total clips planned: {n_total}")
    print(f"Extracted OK:       {n_ok}")
    print(f"Failed:             {n_fail}")
    print(f"Missing sources:    {n_missing_src}")
    print(f"Output clips dir:   {out_dir}")
    print(f"Output manifest:    {out_manifest}")

if __name__ == "__main__":
    main()
