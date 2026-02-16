#!/usr/bin/env python3
import csv
import hashlib
from pathlib import Path

ROOT = Path("/cai3_ds_vm/downloads/deepfake_analysis/DF40")
CELEB_ROOT = ROOT / "video-data" / "celebdf_v2"
TEST_LIST = ROOT / "List_of_testing_videos.txt"
OUT = ROOT / "manifests" / "celebdf_v2_videos_manifest.csv"

# map: folder -> (label_video, source_face, generator)
FOLDERS = {
    "Celeb-real":      ("real", "celebdf", ""),
    "YouTube-real":    ("real", "celebdf_youtube", ""),
    "Celeb-synthesis": ("fake", "celebdf", "celebdf_synthesis"),
}

def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def load_test_keys(test_list_path: Path) -> set[str]:
    """
    Parsează List_of_testing_videos.txt robust.
    Liniile pot arăta ca:
      "1 YouTube-real/00170.mp4"
      "2 Celeb-synthesis/id0_id1_0002.mp4"
    -> noi vrem cheia "YouTube-real/00170.mp4" (subfolder/filename)
    """
    test_keys = set()
    if not test_list_path.exists():
        raise FileNotFoundError(f"Missing test list: {test_list_path}")
    for raw in test_list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()                # separă după whitespace
        path = tokens[-1].replace("\\", "/") # ia ultimul token (robust)
        test_keys.add(path)
    return test_keys

def main():
    test_keys = load_test_keys(TEST_LIST)

    rows = []
    num_mp4 = 0

    for folder_name, (label_video, source_face, generator) in FOLDERS.items():
        folder = CELEB_ROOT / folder_name
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        for p in sorted(folder.glob("*.mp4")):
            num_mp4 += 1
            rel_to_celeb = p.relative_to(CELEB_ROOT).as_posix()  # ex: "YouTube-real/00170.mp4"
            key = rel_to_celeb  # match pe "subfolder/filename"
            split = "test" if key in test_keys else "train"

            # rel_path relativ la DF40 (ca în df40_videos_manifest.csv)
            rel_path = (Path("video-data") / "celebdf_v2" / rel_to_celeb).as_posix()

            # video_id stabil + unic (ca în DF40 manifest)
            video_id = sha1_short(rel_path)

            rows.append({
                "dataset": "CelebDF-v2",
                "split": split,
                "video_id": video_id,
                "rel_path": rel_path,
                "label_video": label_video,
                "generator": generator,
                "source_face": source_face,
                "ext": "mp4",
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "split", "video_id", "rel_path", "label_video", "generator", "source_face", "ext"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # sumar
    counts = {}
    for r in rows:
        k = f"{r['split']}::{r['label_video']}"
        counts[k] = counts.get(k, 0) + 1

    print("Done.")
    print("Celeb root:", str(CELEB_ROOT))
    print("MP4 found:", num_mp4)
    print("Rows written:", len(rows))
    print("Counts split/label:", counts)
    print("Output:", str(OUT))

if __name__ == "__main__":
    main()