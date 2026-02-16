#!/usr/bin/env python3
"""
02_validate_video_streams_df40.py

Adaptare pentru DF40:
- NU scanează organized/train|test.
- Citește manifests/master_videos_manifest.csv (single source of truth) care are abs_path/rel_path/split etc.
- Rulează ffprobe pe abs_path pentru fiecare video.
- Scrie manifestul de stream-uri în: manifests/df40_video_streams_manifest.csv
  (nume nou => nu suprascrie nimic din Deepfake-Eval și nici alte fișiere DF40 existente)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass
class ProbeResult:
    # --- columns from master manifest (kept for traceability / downstream grouping)
    dataset: str
    split: str
    video_id: str
    video_uid: str
    rel_path: str
    abs_path: str
    label_video: str
    generator: str
    source_face: str
    ext: str
    note: str

    # --- probe result
    file_size_bytes: int
    status: str                # video_ok / audio_only / broken
    error_code: str            # "" daca OK
    error_msg: str             # "" daca OK

    # stream properties
    container: str
    has_video: int
    has_audio: int
    video_codec: str
    audio_codec: str
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    duration_s: Optional[float]
    nb_frames: Optional[int]
    pix_fmt: str
    is_vfr: Optional[int]      # 1/0/None

    # derived
    duration_bucket: str


def run_ffprobe(video_path: Path, ffprobe_bin: str) -> Dict[str, Any]:
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    return json.loads(proc.stdout)


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def parse_fps(stream: Dict[str, Any]) -> Optional[float]:
    def frac_to_float(frac: str) -> Optional[float]:
        if not frac or frac == "0/0":
            return None
        if "/" in frac:
            a, b = frac.split("/", 1)
            a_f = safe_float(a)
            b_f = safe_float(b)
            if a_f is None or b_f is None or b_f == 0:
                return None
            return a_f / b_f
        return safe_float(frac)

    avg = frac_to_float(stream.get("avg_frame_rate", ""))
    rfr = frac_to_float(stream.get("r_frame_rate", ""))
    return avg if (avg is not None and avg > 0.1) else rfr


def infer_is_vfr(stream: Dict[str, Any]) -> Optional[int]:
    def frac(fr: str) -> Optional[float]:
        if not fr or fr == "0/0":
            return None
        if "/" in fr:
            a, b = fr.split("/", 1)
            a_f = safe_float(a)
            b_f = safe_float(b)
            if a_f is None or b_f is None or b_f == 0:
                return None
            return a_f / b_f
        return safe_float(fr)

    fps_avg = frac(stream.get("avg_frame_rate", ""))
    fps_r = frac(stream.get("r_frame_rate", ""))

    if fps_avg is None or fps_r is None or fps_avg <= 0 or fps_r <= 0:
        return None

    rel_diff = abs(fps_avg - fps_r) / max(fps_avg, fps_r)
    return 1 if rel_diff > 0.01 else 0


def duration_bucket(duration_s: Optional[float]) -> str:
    if duration_s is None or duration_s <= 0:
        return "unknown"
    if duration_s < 10:
        return "0-10s"
    if duration_s < 30:
        return "10-30s"
    if duration_s < 60:
        return "30-60s"
    if duration_s < 120:
        return "60-120s"
    return "120s+"


def classify_streams(ffp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    streams = ffp.get("streams", []) or []
    fmt = ffp.get("format", {}) or {}

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    has_video = 1 if len(video_streams) > 0 else 0
    has_audio = 1 if len(audio_streams) > 0 else 0

    container = (fmt.get("format_name") or "").split(",")[0] if fmt.get("format_name") else ""

    vcodec = ""
    acodec = ""
    w = h = None
    fps = None
    nb_frames = None
    pix = ""
    is_vfr = None

    if has_video:
        vs = video_streams[0]
        vcodec = vs.get("codec_name") or ""
        w = safe_int(vs.get("width"))
        h = safe_int(vs.get("height"))
        fps = parse_fps(vs)
        nb_frames = safe_int(vs.get("nb_frames"))
        pix = vs.get("pix_fmt") or ""
        is_vfr = infer_is_vfr(vs)

    if has_audio:
        a = audio_streams[0]
        acodec = a.get("codec_name") or ""

    dur = safe_float(fmt.get("duration"))

    if has_video == 1:
        status = "video_ok"
    elif has_audio == 1 and has_video == 0:
        status = "audio_only"
    else:
        status = "broken"

    props = {
        "container": container,
        "has_video": has_video,
        "has_audio": has_audio,
        "video_codec": vcodec or "unknown",
        "audio_codec": acodec or ("unknown" if has_audio else ""),
        "width": w,
        "height": h,
        "fps": fps,
        "duration_s": dur,
        "nb_frames": nb_frames,
        "pix_fmt": pix or "",
        "is_vfr": is_vfr,
    }
    return status, props


def read_master_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def write_manifest_csv(rows: List[ProbeResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))


def main() -> int:
    ap = argparse.ArgumentParser(description="DF40: Validate video streams from master manifest + write streams manifest")
    ap.add_argument("--root", required=True, help="DF40 root folder (contains manifests/ and video-data/)")
    ap.add_argument("--master", default="manifests/master_videos_manifest.csv",
                    help="Master videos manifest (relative to root)")
    ap.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe binary")
    ap.add_argument("--out", default="manifests/df40_video_streams_manifest.csv",
                    help="Output CSV (relative to root)")
    ap.add_argument("--ext", default="mp4,mov,mkv,webm,avi,m4v", help="Comma-separated video extensions to include")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    master_path = root / args.master
    out_csv = root / args.out

    global VIDEO_EXTS
    VIDEO_EXTS = {"." + e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip()}

    if not master_path.exists():
        print(f"[ERROR] master manifest not found: {master_path}")
        return 2

    rows_in = read_master_manifest(master_path)
    all_rows: List[ProbeResult] = []

    for row in rows_in:
        abs_path = Path(row["abs_path"])
        if abs_path.suffix.lower() not in VIDEO_EXTS:
            continue  # ignore non-video

        size = abs_path.stat().st_size if abs_path.exists() else -1

        try:
            ffp = run_ffprobe(abs_path, args.ffprobe)
            status, props = classify_streams(ffp)
            bucket = duration_bucket(props.get("duration_s"))

            all_rows.append(
                ProbeResult(
                    dataset=row.get("dataset", ""),
                    split=row.get("split", ""),
                    video_id=row.get("video_id", ""),
                    video_uid=row.get("video_uid", ""),
                    rel_path=row.get("rel_path", ""),
                    abs_path=row.get("abs_path", ""),
                    label_video=row.get("label_video", ""),
                    generator=row.get("generator", ""),
                    source_face=row.get("source_face", ""),
                    ext=row.get("ext", abs_path.suffix.lower().lstrip(".")),
                    note=row.get("note", ""),

                    file_size_bytes=int(size),
                    status=status,
                    error_code="",
                    error_msg="",

                    container=props.get("container", ""),
                    has_video=int(props.get("has_video", 0)),
                    has_audio=int(props.get("has_audio", 0)),
                    video_codec=str(props.get("video_codec", "unknown")),
                    audio_codec=str(props.get("audio_codec", "")),
                    width=props.get("width"),
                    height=props.get("height"),
                    fps=props.get("fps"),
                    duration_s=props.get("duration_s"),
                    nb_frames=props.get("nb_frames"),
                    pix_fmt=str(props.get("pix_fmt", "")),
                    is_vfr=props.get("is_vfr"),

                    duration_bucket=bucket,
                )
            )

        except Exception as e:
            # broken: keep row + error
            all_rows.append(
                ProbeResult(
                    dataset=row.get("dataset", ""),
                    split=row.get("split", ""),
                    video_id=row.get("video_id", ""),
                    video_uid=row.get("video_uid", ""),
                    rel_path=row.get("rel_path", ""),
                    abs_path=row.get("abs_path", ""),
                    label_video=row.get("label_video", ""),
                    generator=row.get("generator", ""),
                    source_face=row.get("source_face", ""),
                    ext=row.get("ext", abs_path.suffix.lower().lstrip(".")),
                    note=row.get("note", ""),

                    file_size_bytes=int(size),
                    status="broken",
                    error_code="FFPROBE_FAIL",
                    error_msg=str(e)[:500],

                    container="",
                    has_video=0,
                    has_audio=0,
                    video_codec="unknown",
                    audio_codec="",
                    width=None,
                    height=None,
                    fps=None,
                    duration_s=None,
                    nb_frames=None,
                    pix_fmt="",
                    is_vfr=None,

                    duration_bucket="unknown",
                )
            )

    if not all_rows:
        print("[WARN] no rows written (no videos matched ext filter?)")
        return 3

    write_manifest_csv(all_rows, out_csv)

    # Summary
    counts: Dict[str, int] = {}
    for r in all_rows:
        counts[r.status] = counts.get(r.status, 0) + 1

    print(f"[OK] Wrote: {out_csv}")
    print("[SUMMARY] status counts:", counts)
    print("[SUMMARY] total rows:", len(all_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
