#!/usr/bin/env python3
"""
02_validate_video_streams.py

Scop:
- Parcurge organized/train si organized/test
- Pentru fiecare .mp4 (sau extensii video) face sanity-check pe stream-uri cu ffprobe:
    - are video stream?
    - are audio stream?
    - codec-uri, rezolutie, fps, durata
- Eticheteaza status: video_ok / audio_only / broken
- (optional) face triere in subfoldere: organized/{split}/{status}/...
- Scrie un manifest CSV "single source of truth" in manifests/video_streams_manifest.csv

Cerințe:
- ffprobe instalat (din ffmpeg)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass
class ProbeResult:
    split: str                 # train / test
    filename: str              # ex: abc.mp4
    video_id: str              # nume fara extensie
    rel_path: str              # path relativ la root dataset
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
    is_vfr: Optional[int]      # 1/0/None (heuristic)

    # derived signals
    duration_bucket: str

    # mapping after triere
    new_rel_path: str          # "" daca nu s-a triat
    moved: int                 # 1 daca s-a mutat/copiat, altfel 0


def run_ffprobe(video_path: Path, ffprobe_bin: str) -> Dict[str, Any]:
    """
    Rulează ffprobe și întoarce JSON-ul brut.
    Dacă ffprobe eșuează (fișier corupt / container invalid), aruncă excepție.
    """
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
    """
    fps poate veni ca 'r_frame_rate' sau 'avg_frame_rate' (ex: "30000/1001").
    Preferăm avg_frame_rate (mai aproape de real), fallback la r_frame_rate.
    """
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
    # dacă avg e foarte mic/None, ia r_frame_rate
    return avg if (avg is not None and avg > 0.1) else rfr


def infer_is_vfr(stream: Dict[str, Any]) -> Optional[int]:
    """
    Heuristic: dacă avg_frame_rate diferă semnificativ de r_frame_rate => probabil VFR.
    Nu e perfect, dar suficient ca semnal.
    """
    fps_avg = None
    fps_r = None

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

    # diferență relativă > 1% => suspect VFR
    rel_diff = abs(fps_avg - fps_r) / max(fps_avg, fps_r)
    return 1 if rel_diff > 0.01 else 0


def duration_bucket(duration_s: Optional[float]) -> str:
    """
    Binning simplu pentru decizii ulterioare de clip-splitting.
    """
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
    """
    Ia JSON-ul ffprobe și decide:
    - video_ok: are video stream valid
    - audio_only: are audio dar nu are video
    - broken: nimic util / inconsistent

    Întoarce status + un dict cu proprietăți extrase.
    """
    streams = ffp.get("streams", []) or []
    fmt = ffp.get("format", {}) or {}

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    has_video = 1 if len(video_streams) > 0 else 0
    has_audio = 1 if len(audio_streams) > 0 else 0

    container = (fmt.get("format_name") or "").split(",")[0] if fmt.get("format_name") else ""

    # extragem din primul video stream (suficient pentru sanity)
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

    # clasificare
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


def iter_media_files(root: Path, split: str) -> List[Path]:
    """
    Listează fișierele video din organized/{split}, recursiv.
    Notă: dacă ai deja subfoldere, tot le prinde.
    """
    base = root / "organized" / split
    if not base.exists():
        return []
    files: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    return sorted(files)


def compute_rel_path(root: Path, abs_path: Path) -> str:
    return str(abs_path.resolve().relative_to(root.resolve()))


def triage_file(
    root: Path,
    abs_path: Path,
    split: str,
    status: str,
    mode: str,
) -> Tuple[str, int]:
    """
    Dacă mode == "none": nu mută nimic.
    Dacă mode == "copy" / "move": pune fișierul în:
      organized/{split}/{status}/{filename}

    Întoarce new_rel_path și moved flag (1/0).
    """
    if mode == "none":
        return "", 0

    dest_dir = root / "organized" / split / status
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / abs_path.name

    # Evităm overwrite: dacă există deja, păstrăm originalul și nu mutăm.
    if dest_path.exists():
        return compute_rel_path(root, dest_path), 0

    if mode == "copy":
        shutil.copy2(abs_path, dest_path)
        return compute_rel_path(root, dest_path), 1

    if mode == "move":
        shutil.move(str(abs_path), str(dest_path))
        return compute_rel_path(root, dest_path), 1

    raise ValueError(f"Unknown triage mode: {mode}")


def build_probe_result(
    root: Path,
    split: str,
    file_path: Path,
    status: str,
    props: Dict[str, Any],
    error_code: str = "",
    error_msg: str = "",
    new_rel_path: str = "",
    moved: int = 0,
) -> ProbeResult:
    vid = file_path.stem
    rel = compute_rel_path(root, file_path)
    size = file_path.stat().st_size if file_path.exists() else -1
    dur = props.get("duration_s")
    bucket = duration_bucket(dur)

    return ProbeResult(
        split=split,
        filename=file_path.name,
        video_id=vid,
        rel_path=rel,
        file_size_bytes=size,

        status=status,
        error_code=error_code,
        error_msg=error_msg,

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

        new_rel_path=new_rel_path,
        moved=int(moved),
    )


def write_manifest_csv(rows: List[ProbeResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate video streams + duration + triage + manifest")
    ap.add_argument("--root", required=True, help="Dataset root folder (the folder that contains organized/)")
    ap.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe binary (default: ffprobe in PATH)")
    ap.add_argument("--triage", choices=["none", "copy", "move"], default="none",
                    help="If set, copy/move files into organized/{split}/{status}/")
    ap.add_argument("--out", default="manifests/video_streams_manifest.csv",
                    help="Output manifest CSV path (relative to root)")
    ap.add_argument("--ext", default="mp4,mov,mkv,webm,avi,m4v", help="Comma-separated video extensions to include")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = root / args.out

    global VIDEO_EXTS
    VIDEO_EXTS = {"." + e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip()}

    all_rows: List[ProbeResult] = []

    for split in ["train", "test"]:
        files = iter_media_files(root, split)
        if not files:
            print(f"[WARN] No files found under {root}/organized/{split}")
            continue

        for fp in files:
            try:
                ffp = run_ffprobe(fp, args.ffprobe)
                status, props = classify_streams(ffp)

                new_rel = ""
                moved = 0
                if args.triage != "none":
                    new_rel, moved = triage_file(root, fp, split, status, args.triage)

                row = build_probe_result(
                    root=root,
                    split=split,
                    file_path=fp,
                    status=status,
                    props=props,
                    new_rel_path=new_rel,
                    moved=moved,
                )
                all_rows.append(row)

            except Exception as e:
                # broken: nu putem proba; păstrăm eroarea
                props = {
                    "container": "",
                    "has_video": 0,
                    "has_audio": 0,
                    "video_codec": "unknown",
                    "audio_codec": "",
                    "width": None,
                    "height": None,
                    "fps": None,
                    "duration_s": None,
                    "nb_frames": None,
                    "pix_fmt": "",
                    "is_vfr": None,
                }
                row = build_probe_result(
                    root=root,
                    split=split,
                    file_path=fp,
                    status="broken",
                    props=props,
                    error_code="FFPROBE_FAIL",
                    error_msg=str(e)[:500],
                )
                all_rows.append(row)

    if all_rows:
        write_manifest_csv(all_rows, out_csv)
        print(f"[OK] Wrote manifest: {out_csv}")
        # sumar scurt
        counts: Dict[str, int] = {}
        for r in all_rows:
            counts[r.status] = counts.get(r.status, 0) + 1
        print("[SUMMARY] status counts:", counts)
    else:
        print("[WARN] No rows to write. Check your --root and folder structure.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
