# scripts/04_merge_df40_and_celebdf.py
from __future__ import annotations
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # DF40/
DF40_MAN = ROOT / "manifests/df40_videos_manifest.csv"
CELEB_MAN = ROOT / "manifests/celebdf_v2_videos_manifest.csv"
OUT_MAN = ROOT / "manifests/master_videos_manifest.csv"

CORE_COLS = [
    "dataset",
    "split",
    "video_id",
    "video_uid",
    "rel_path",
    "abs_path",
    "label_video",
    "generator",
    "source_face",
    "ext",
    "note",
]

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def read_rows(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)

def normalize(rows: list[dict], root: Path) -> list[dict]:
    out = []
    for row in rows:
        rel = (row.get("rel_path") or "").strip()
        # unele manifesturi au rel_path fără prefix "video-data/..."? păstrăm cum e, dar trebuie să fie relativ la ROOT
        if not rel:
            continue

        # video_uid (dacă lipsește)
        if not (row.get("video_uid") or "").strip():
            row["video_uid"] = sha1(rel)

        # abs_path (dacă lipsește sau e gol)
        if not (row.get("abs_path") or "").strip():
            row["abs_path"] = str((root / rel).resolve())

        # ext (dacă lipsește)
        if not (row.get("ext") or "").strip():
            row["ext"] = Path(rel).suffix.lstrip(".").lower()

        # note (dacă lipsește)
        if "note" not in row:
            row["note"] = ""

        # păstrăm doar coloanele core + orice extra din original nu ne trebuie la merge
        out.append({k: (row.get(k, "") or "") for k in CORE_COLS})
    return out

def write_rows(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CORE_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    df40 = normalize(read_rows(DF40_MAN), ROOT)
    celeb = normalize(read_rows(CELEB_MAN), ROOT)

    merged = df40 + celeb
    write_rows(OUT_MAN, merged)

    print("Done.")
    print("DF40 rows:", len(df40))
    print("CelebDF rows:", len(celeb))
    print("Merged rows:", len(merged))
    print("Output:", OUT_MAN)

if __name__ == "__main__":
    main()
