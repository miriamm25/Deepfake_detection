#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List


@dataclass
class CropDetectResult:
    w: Optional[int] = None
    h: Optional[int] = None
    x: Optional[int] = None
    y: Optional[int] = None
    raw: str = ""


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    rows = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def duration_bucket(seconds: float) -> str:
    if seconds < 10:
        return "0-10s"
    if seconds < 30:
        return "10-30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    return "120s+"


def cropdetect_ffmpeg(ffmpeg: str, video_path: Path, probe_seconds: float = 1.0) -> CropDetectResult:
    """
    Uses ffmpeg cropdetect to guess active content area.
    We DO NOT apply this crop here; we only log/flag it.

    Parses last occurrence of 'crop=w:h:x:y' from stderr.
    """
    cmd = [
        ffmpeg, "-hide_banner",
        "-ss", "0", "-t", str(probe_seconds),
        "-i", str(video_path),
        "-vf", "cropdetect=24:16:0",
        "-f", "null", "-"
    ]
    rc, out, err = run_cmd(cmd)
    text = err or ""

    m = None
    for mm in re.finditer(r"crop=(\d+):(\d+):(\d+):(\d+)", text):
        m = mm
    if not m:
        return CropDetectResult(raw=text[-2000:])

    w, h, x, y = map(int, m.groups())
    return CropDetectResult(w=w, h=h, x=x, y=y, raw=text[-2000:])


def suspected_black_bars(
    width: int,
    height: int,
    crop: CropDetectResult,
    min_border_px: int = 8,
    min_area_ratio: float = 0.92
) -> bool:
    """
    Heuristic:
    - if cropdetect suggests a noticeably smaller active area than full frame,
      flag as suspected bars / padding.
    """
    if not crop.w or not crop.h or width <= 0 or height <= 0:
        return False

    border_x = width - crop.w
    border_y = height - crop.h

    if border_x < min_border_px and border_y < min_border_px:
        return False

    area_total = width * height
    area_crop = crop.w * crop.h
    if area_total <= 0:
        return False

    ratio = area_crop / area_total
    return ratio < min_area_ratio


def ffprobe_basic(ffprobe: str, video_path: Path) -> Dict[str, object]:
    """
    Returns basic stream+format info needed for:
    - black bar suspicion
    - validation (resume/skip)
    """
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,codec_name,pix_fmt,avg_frame_rate,r_frame_rate"
        ":format=duration",
        "-of", "json",
        str(video_path)
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0 or not out.strip():
        raise RuntimeError((err or out or "ffprobe failed")[-2000:])

    j = json.loads(out)
    streams = j.get("streams") or []
    fmt = j.get("format") or {}
    s0 = streams[0] if streams else {}

    def parse_rate(x: str) -> float:
        try:
            if not x or x == "0/0":
                return 0.0
            a, b = x.split("/")
            return float(a) / float(b)
        except Exception:
            return 0.0

    width = int(s0.get("width") or 0)
    height = int(s0.get("height") or 0)
    codec = (s0.get("codec_name") or "").strip()
    pix_fmt = (s0.get("pix_fmt") or "").strip()
    r_fps = parse_rate(s0.get("r_frame_rate") or "")
    avg_fps = parse_rate(s0.get("avg_frame_rate") or "")

    try:
        duration = float(fmt.get("duration") or 0.0)
    except Exception:
        duration = 0.0

    return {
        "width": width,
        "height": height,
        "codec": codec,
        "pix_fmt": pix_fmt,
        "r_fps": r_fps,
        "avg_fps": avg_fps,
        "duration": duration,
    }


def is_std_ok(
    ffprobe: str,
    p: Path,
    target_size: int,
    target_fps: int,
    target_pix: str,
    target_codec: str,
    target_dur: float
) -> bool:
    """
    Industry-ish resume check:
    - 1024x1024
    - codec hevc
    - pix_fmt yuv420p10le
    - duration ~5.0s (tolerance)
    - fps ~24 (tolerance)
    """
    try:
        m = ffprobe_basic(ffprobe, p)
        if int(m["width"]) != target_size or int(m["height"]) != target_size:
            return False
        if (m["codec"] or "").lower() != target_codec:
            return False
        if (m["pix_fmt"] or "") != target_pix:
            return False
        if abs(float(m["duration"]) - float(target_dur)) > 0.08:
            return False
        fps = float(m["avg_fps"] or m["r_fps"] or 0.0)
        if abs(fps - float(target_fps)) > 0.5:
            return False
        return True
    except Exception:
        return False


def ffmpeg_standardize_clip(
    ffmpeg: str,
    src_path: Path,
    dst_path: Path,
    clip_len_s: float,
    fps_out: int,
    size_out: int,
    remove_audio: bool,
    crf: int,
    preset: str
) -> Tuple[int, str]:
    ensure_dir(dst_path.parent)

    vf = (
        "crop=min(iw\\,ih):min(iw\\,ih):(iw-min(iw\\,ih))/2:(ih-min(iw\\,ih))/2,"
        f"scale={size_out}:{size_out},fps={fps_out}"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", str(src_path),
        "-t", str(clip_len_s),
        "-vf", vf,
        "-c:v", "libx265",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p10le",
        "-tag:v", "hvc1",
    ]
    if remove_audio:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    cmd += [str(dst_path)]

    rc, out, err = run_cmd(cmd)
    msg = (err or out or "").strip()
    return rc, msg[-2000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repo root (e.g., /.../Deepfake-Eval-2024)")
    ap.add_argument("--manifest", default="manifests/clips_on_disk_manifest.csv", help="Input clips manifest (relative to root)")
    ap.add_argument("--out_dir", default="clips_std", help="Output standardized clips dir (relative to root)")
    ap.add_argument("--out_manifest", default="manifests/clips_std_manifest.csv", help="Output manifest path (relative to root)")
    ap.add_argument("--report_dir", default="reports/clips_std", help="Report dir (relative to root)")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary")
    ap.add_argument("--ffprobe", default="ffprobe", help="ffprobe binary")
    ap.add_argument("--clip_len_s", type=float, default=5.0, help="Target clip duration in seconds")
    ap.add_argument("--fps", type=int, default=24, help="Target CFR FPS")
    ap.add_argument("--size", type=int, default=1024, help="Target square resolution")
    ap.add_argument("--remove_audio", action="store_true", help="Strip audio from standardized clips")
    ap.add_argument("--crf", type=int, default=18, help="x265 CRF (lower=better quality, larger files)")
    ap.add_argument("--preset", default="medium", help="x265 preset")
    ap.add_argument("--detect_black_bars", action="store_true", help="Run cropdetect and flag suspected black bars (no cropping changes)")
    ap.add_argument("--dry_run", action="store_true", help="Plan only, do not run ffmpeg (still can run ffprobe/cropdetect)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest_in = root / args.manifest
    out_dir = root / args.out_dir
    manifest_out = root / args.out_manifest
    report_dir = root / args.report_dir
    ensure_dir(report_dir)

    if not manifest_in.exists():
        raise FileNotFoundError(f"Input manifest not found: {manifest_in}")

    rows = read_manifest_rows(manifest_in)

    out_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []

    total = 0
    ok = 0
    skipped_exists_ok = 0
    skipped_non_ok_status = 0

    for r in rows:
        # (1) status filter: procesează doar status == ok
        status = (r.get("status") or "").strip().lower()
        if status and status != "ok":
            skipped_non_ok_status += 1
            continue

        split = (r.get("split") or "train").strip()
        video_id = (r.get("video_id") or "unknown").strip()

        # În manifestul tău, clipul deja extras e în clip_rel_path
        src_rel = (r.get("clip_rel_path") or r.get("clip_abs_path") or "").strip()
        if not src_rel:
            continue

        # resolve src path
        if os.path.isabs(src_rel):
            src_path = Path(src_rel)
        else:
            src_path = (root / src_rel).resolve()

        clip_id = (r.get("clip_id") or src_path.stem).strip()

        if not src_path.exists():
            failed_rows.append({
                "split": split, "video_id": video_id, "clip_id": clip_id,
                "src_rel_path": src_rel, "error": "missing_source"
            })
            continue

        # Build destination path
        out_rel = f"{args.out_dir}/{split}/{video_id}/{clip_id}.mp4"
        out_path = (root / out_rel).resolve()

        # (2) meta from ffprobe (source)
        try:
            meta_src = ffprobe_basic(args.ffprobe, src_path)
        except Exception as e:
            failed_rows.append({
                "split": split, "video_id": video_id, "clip_id": clip_id,
                "src_rel_path": src_rel, "error": "ffprobe_failed", "msg": str(e)[:2000]
            })
            continue

        width = int(meta_src["width"])
        height = int(meta_src["height"])
        fps_src = float(meta_src["avg_fps"] or meta_src["r_fps"] or 0.0)
        dur_src = float(meta_src["duration"] or 0.0)
        vcodec = (meta_src["codec"] or "")
        pix_fmt = (meta_src["pix_fmt"] or "")

        # optional: cropdetect for bars flag (runs also in dry_run now)
        crop_res = CropDetectResult()
        bars_flag = 0
        if args.detect_black_bars:
            crop_res = cropdetect_ffmpeg(args.ffmpeg, src_path, probe_seconds=1.0)
            bars_flag = 1 if suspected_black_bars(width, height, crop_res) else 0

        # (3) resume/skip if output exists and is already valid standardized clip
        if out_path.exists() and out_path.stat().st_size > 0:
            if is_std_ok(args.ffprobe, out_path, args.size, args.fps, "yuv420p10le", "hevc", args.clip_len_s):
                total += 1
                ok += 1
                skipped_exists_ok += 1

                out_rows.append({
                    "split": split,
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "src_rel_path": src_rel,
                    "dst_rel_path": out_rel,
                    "duration_s_src": dur_src,
                    "fps_src": fps_src,
                    "width_src": width,
                    "height_src": height,
                    "video_codec_src": vcodec,
                    "pix_fmt_src": pix_fmt,
                    "duration_bucket_src": duration_bucket(dur_src) if dur_src > 0 else "",
                    "target_clip_len_s": args.clip_len_s,
                    "target_fps": args.fps,
                    "target_size": args.size,
                    "target_codec": "hevc_x265",
                    "target_pix_fmt": "yuv420p10le",
                    "remove_audio": 1 if args.remove_audio else 0,
                    "suspected_black_bars": bars_flag,
                    "cropdetect_w": crop_res.w if crop_res.w is not None else "",
                    "cropdetect_h": crop_res.h if crop_res.h is not None else "",
                    "cropdetect_x": crop_res.x if crop_res.x is not None else "",
                    "cropdetect_y": crop_res.y if crop_res.y is not None else "",
                    "ffmpeg_rc": 0,
                    "ffmpeg_msg": "skipped_exists_ok",
                })
                continue
            # dacă există dar nu e ok, îl recodăm (ffmpeg -y va suprascrie)

        total += 1

        if args.dry_run:
            rc = 0
            msg = ""
        else:
            rc, msg = ffmpeg_standardize_clip(
                ffmpeg=args.ffmpeg,
                src_path=src_path,
                dst_path=out_path,
                clip_len_s=args.clip_len_s,
                fps_out=args.fps,
                size_out=args.size,
                remove_audio=args.remove_audio,
                crf=args.crf,
                preset=args.preset,
            )

        out_row = {
            "split": split,
            "video_id": video_id,
            "clip_id": clip_id,
            "src_rel_path": src_rel,
            "dst_rel_path": out_rel,
            "duration_s_src": dur_src,
            "fps_src": fps_src,
            "width_src": width,
            "height_src": height,
            "video_codec_src": vcodec,
            "pix_fmt_src": pix_fmt,
            "duration_bucket_src": duration_bucket(dur_src) if dur_src > 0 else "",
            "target_clip_len_s": args.clip_len_s,
            "target_fps": args.fps,
            "target_size": args.size,
            "target_codec": "hevc_x265",
            "target_pix_fmt": "yuv420p10le",
            "remove_audio": 1 if args.remove_audio else 0,
            "suspected_black_bars": bars_flag,
            "cropdetect_w": crop_res.w if crop_res.w is not None else "",
            "cropdetect_h": crop_res.h if crop_res.h is not None else "",
            "cropdetect_x": crop_res.x if crop_res.x is not None else "",
            "cropdetect_y": crop_res.y if crop_res.y is not None else "",
            "ffmpeg_rc": rc,
            "ffmpeg_msg": "" if rc == 0 else msg,
        }
        out_rows.append(out_row)

        if rc == 0:
            ok += 1
        else:
            failed_rows.append({
                "split": split, "video_id": video_id, "clip_id": clip_id,
                "src_rel_path": src_rel, "dst_rel_path": out_rel,
                "ffmpeg_rc": rc, "ffmpeg_msg": msg
            })

    # Write outputs (even if partially processed; that's the whole point of resumable)
    out_fields = list(out_rows[0].keys()) if out_rows else []
    write_csv(manifest_out, out_fields, out_rows)

    if failed_rows:
        failed_path = report_dir / "failed.csv"
        failed_fields = list(failed_rows[0].keys())
        write_csv(failed_path, failed_fields, failed_rows)

    summary = {
        "input_manifest": str(Path(args.manifest)),
        "output_manifest": str(Path(args.out_manifest)),
        "output_clips_dir": str(Path(args.out_dir)),
        "rows_in": len(rows),
        "rows_emitted": len(out_rows),
        "rows_processed": total,
        "standardized_ok": ok,
        "failed": len(failed_rows),
        "skipped_non_ok_status": skipped_non_ok_status,
        "skipped_exists_ok": skipped_exists_ok,
        "settings": {
            "clip_len_s": args.clip_len_s,
            "fps": args.fps,
            "size": args.size,
            "codec": "libx265",
            "pix_fmt": "yuv420p10le",
            "remove_audio": bool(args.remove_audio),
            "crf": args.crf,
            "preset": args.preset,
            "detect_black_bars": bool(args.detect_black_bars),
            "dry_run": bool(args.dry_run),
        },
    }

    if args.detect_black_bars:
        summary["suspected_black_bars_count"] = sum(int(rr.get("suspected_black_bars", 0) or 0) for rr in out_rows)

    with (report_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Processed clips: {total}")
    print(f"Standardized OK: {ok}")
    print(f"Failed: {len(failed_rows)}")
    print(f"Skipped (non-ok status): {skipped_non_ok_status}")
    print(f"Skipped (exists & already std): {skipped_exists_ok}")
    print(f"Output clips dir: {out_dir}")
    print(f"Output manifest: {manifest_out}")
    print(f"Report dir: {report_dir}")


if __name__ == "__main__":
    main()
