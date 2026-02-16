#!/usr/bin/env python3
import argparse
import csv
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Row:
    dataset: str
    split: str
    video_id: str          # stem (nume fișier fără extensie)
    video_uid: str         # id stabil unic (hash din path relativ)
    rel_path: str          # ex: video-data/test/heygen/fake/abc.mp4
    abs_path: str
    label_video: str       # real/fake/unknown
    generator: str         # folder după split
    source_face: str       # cdf/ff/""
    ext: str               # mp4
    note: str              # pentru debugging (opțional)


def sha1_short(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def infer_split_and_generator(rel_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    rel_path: video-data/<split>/<generator>/...
    """
    parts = Path(rel_path).parts
    # parts[0] = "video-data"
    if len(parts) < 3 or parts[0] != "video-data":
        return None, None
    split = parts[1]
    generator = parts[2]
    if split not in ("train", "test"):
        return None, None
    return split, generator


def infer_source_face(rel_path: str) -> str:
    # dacă apare "cdf" sau "ff" ca segment de folder în path
    parts = set(Path(rel_path).parts)
    if "cdf" in parts:
        return "cdf"
    if "ff" in parts:
        return "ff"
    return ""


def infer_label(rel_path: str, default_label: str = "fake") -> str:
    """
    Reguli:
    - dacă path conține /real/ sau /real_mp4/ -> real
    - dacă path conține /fake/ -> fake
    - altfel -> default_label (în DF40, de obicei fake)
    """
    parts = Path(rel_path).parts

    # prioritizez "real" dacă apare explicit
    if "real_mp4" in parts or "real" in parts:
        return "real"
    if "fake" in parts:
        return "fake"
    return default_label


def iter_videos(video_root: Path, exts: Tuple[str, ...] = (".mp4",)) -> List[Path]:
    vids: List[Path] = []
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            vids.append(p)
    return vids


def write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "split", "video_id", "video_uid",
        "rel_path", "abs_path",
        "label_video", "generator", "source_face",
        "ext", "note"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r.dataset,
                "split": r.split,
                "video_id": r.video_id,
                "video_uid": r.video_uid,
                "rel_path": r.rel_path,
                "abs_path": r.abs_path,
                "label_video": r.label_video,
                "generator": r.generator,
                "source_face": r.source_face,
                "ext": r.ext,
                "note": r.note,
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path către DF40 (ex: /.../deepfake_analysis/DF40)")
    ap.add_argument("--video_dir", default="video-data", help="Folder video relativ la root (default: video-data)")
    ap.add_argument("--out_manifest", default="manifests/df40_videos_manifest.csv", help="CSV output relativ la root")
    ap.add_argument("--report_dir", default="reports/df40_manifest", help="Report dir relativ la root")
    ap.add_argument("--dataset_name", default="DF40", help="Valoare pentru coloana dataset")
    ap.add_argument("--default_label", default="fake", choices=["fake", "unknown"], help="Label default dacă nu există real/fake în path")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    video_root = (root / args.video_dir).resolve()
    out_manifest = (root / args.out_manifest).resolve()
    report_dir = (root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not video_root.exists():
        raise FileNotFoundError(f"Nu există video root: {video_root}")

    files = iter_videos(video_root, exts=(".mp4",))
    rows: List[Row] = []
    skipped: List[Dict[str, str]] = []

    for f in files:
        rel_path = f.relative_to(root).as_posix()
        split, generator = infer_split_and_generator(rel_path)
        if split is None or generator is None:
            skipped.append({"path": rel_path, "reason": "not_under_video-data/<train|test>/<generator>/"})
            continue

        label = infer_label(rel_path, default_label=args.default_label)
        source_face = infer_source_face(rel_path)
        video_id = f.stem
        video_uid = sha1_short(rel_path)

        rows.append(Row(
            dataset=args.dataset_name,
            split=split,
            video_id=video_id,
            video_uid=video_uid,
            rel_path=rel_path,
            abs_path=str(f.resolve()),
            label_video=label,
            generator=generator,
            source_face=source_face,
            ext=f.suffix.lower().lstrip("."),
            note=""
        ))

    # Stats
    counts: Dict[str, int] = {}
    for r in rows:
        key = f"{r.split}::{r.label_video}"
        counts[key] = counts.get(key, 0) + 1

    by_gen: Dict[str, int] = {}
    for r in rows:
        k = f"{r.split}::{r.generator}::{r.label_video}"
        by_gen[k] = by_gen.get(k, 0) + 1

    # Write outputs
    write_csv(out_manifest, rows)

    (report_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    summary = {
        "root": str(root),
        "video_root": str(video_root),
        "out_manifest": str(out_manifest),
        "num_mp4_found_under_video_root": len(files),
        "num_rows_written": len(rows),
        "num_skipped": len(skipped),
        "counts_split_label": counts,
        "counts_split_generator_label_top20": dict(sorted(by_gen.items(), key=lambda x: -x[1])[:20]),
        "default_label_when_ambiguous": args.default_label,
        "notes": [
            "label_video e inferat din path: real/real_mp4 vs fake; altfel default_label",
            "generator = folderul imediat după train/test",
            "source_face = cdf/ff dacă apare ca segment în path",
            "video_uid = sha1(rel_path) -> stabil și unic (evită coliziuni dacă există același nume în foldere diferite)"
        ]
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"MP4 found: {len(files)}")
    print(f"Rows written: {len(rows)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Output manifest: {out_manifest}")
    print(f"Report dir: {report_dir}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
