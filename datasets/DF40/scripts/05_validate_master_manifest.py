#!/usr/bin/env python3
"""
Validates master_videos_manifest.csv by checking:
1. File existence (abs_path)
2. Path consistency (rel_path vs abs_path)
3. Field consistency with path structure (split, generator, label_video, source_face)
4. Logical consistency (video_id matches filename, ext matches extension)
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]  # DF40/
MANIFEST = ROOT / "manifests/master_videos_manifest.csv"
REPORT_DIR = ROOT / "reports/manifest_validation"


def infer_split_and_generator(rel_path: str, dataset: str) -> tuple[Optional[str], Optional[str]]:
    """Extract split and generator from rel_path."""
    parts = Path(rel_path).parts
    if len(parts) < 3 or parts[0] != "video-data":
        return None, None
    
    # CelebDF-v2 has different structure: video-data/celebdf_v2/<folder>/<file>
    if dataset == "CelebDF-v2":
        if parts[1] == "celebdf_v2" and len(parts) >= 3:
            # Split is determined by test list, not path, so we can't infer it
            # Generator is empty for Celeb-real/YouTube-real, or "celebdf_synthesis" for Celeb-synthesis
            folder = parts[2]
            if folder == "Celeb-synthesis":
                return None, "celebdf_synthesis"
            else:
                return None, ""  # Empty generator is valid for CelebDF-v2
        return None, None
    
    # DF40 structure: video-data/<split>/<generator>/...
    split = parts[1]
    generator = parts[2]
    if split not in ("train", "test"):
        return None, None
    return split, generator


def infer_source_face(rel_path: str, dataset: str) -> str:
    """Extract source_face from rel_path."""
    parts = Path(rel_path).parts
    
    # CelebDF-v2 has special source_face values
    if dataset == "CelebDF-v2":
        if "Celeb-real" in parts:
            return "celebdf"
        if "YouTube-real" in parts:
            return "celebdf_youtube"
        if "Celeb-synthesis" in parts:
            return "celebdf"
        return ""
    
    # DF40: check for cdf or ff in path
    parts_set = set(parts)
    if "cdf" in parts_set:
        return "cdf"
    if "ff" in parts_set:
        return "ff"
    return ""


def infer_label(rel_path: str) -> str:
    """Extract label_video from rel_path."""
    parts = Path(rel_path).parts
    if "real_mp4" in parts or "real" in parts:
        return "real"
    if "fake" in parts:
        return "fake"
    # For CelebDF-v2, check folder names
    if "Celeb-real" in parts or "YouTube-real" in parts:
        return "real"
    if "Celeb-synthesis" in parts:
        return "fake"
    # Default: unknown (but we'll flag this as a potential issue)
    return "unknown"


def validate_row(row: Dict[str, str], row_num: int, root: Path) -> List[Dict[str, str]]:
    """Validate a single row and return list of errors."""
    errors = []
    
    dataset = (row.get("dataset") or "").strip()
    rel_path = (row.get("rel_path") or "").strip()
    abs_path = (row.get("abs_path") or "").strip()
    split = (row.get("split") or "").strip()
    generator = (row.get("generator") or "").strip()
    label_video = (row.get("label_video") or "").strip()
    source_face = (row.get("source_face") or "").strip()
    video_id = (row.get("video_id") or "").strip()
    ext = (row.get("ext") or "").strip()
    
    # 1. Check if abs_path exists
    if abs_path:
        abs_file = Path(abs_path)
        if not abs_file.exists():
            errors.append({
                "row": row_num,
                "type": "file_missing",
                "field": "abs_path",
                "value": abs_path,
                "message": f"File does not exist: {abs_path}"
            })
        elif not abs_file.is_file():
            errors.append({
                "row": row_num,
                "type": "not_a_file",
                "field": "abs_path",
                "value": abs_path,
                "message": f"Path exists but is not a file: {abs_path}"
            })
    
    # 2. Check rel_path consistency with abs_path
    if rel_path and abs_path:
        expected_abs = (root / rel_path).resolve()
        actual_abs = Path(abs_path).resolve()
        if expected_abs != actual_abs:
            errors.append({
                "row": row_num,
                "type": "path_mismatch",
                "field": "rel_path/abs_path",
                "value": f"rel={rel_path}, abs={abs_path}",
                "message": f"rel_path does not match abs_path. Expected: {expected_abs}, Got: {actual_abs}"
            })
    
    # 3. Check split and generator consistency with path
    if rel_path:
        inferred_split, inferred_generator = infer_split_and_generator(rel_path, dataset)
        
        # For CelebDF-v2, split is determined by test list, not path, so we skip split validation
        if dataset == "CelebDF-v2":
            # Only validate generator for CelebDF-v2
            if inferred_generator is not None:
                if generator != inferred_generator:
                    errors.append({
                        "row": row_num,
                        "type": "generator_mismatch",
                        "field": "generator",
                        "value": generator,
                        "message": f"generator field '{generator}' does not match path (expected '{inferred_generator}')"
                    })
        else:
            # DF40: validate both split and generator
            if inferred_split is None:
                errors.append({
                    "row": row_num,
                    "type": "invalid_path_structure",
                    "field": "rel_path",
                    "value": rel_path,
                    "message": "rel_path does not follow video-data/<split>/<generator>/... structure"
                })
            else:
                if split and split != inferred_split:
                    errors.append({
                        "row": row_num,
                        "type": "split_mismatch",
                        "field": "split",
                        "value": split,
                        "message": f"split field '{split}' does not match path (expected '{inferred_split}')"
                    })
                
                if inferred_generator:
                    if generator and generator != inferred_generator:
                        errors.append({
                            "row": row_num,
                            "type": "generator_mismatch",
                            "field": "generator",
                            "value": generator,
                            "message": f"generator field '{generator}' does not match path (expected '{inferred_generator}')"
                        })
    
    # 4. Check label_video consistency with path
    if rel_path:
        inferred_label = infer_label(rel_path)
        if label_video and inferred_label != "unknown" and label_video != inferred_label:
            errors.append({
                "row": row_num,
                "type": "label_mismatch",
                "field": "label_video",
                "value": label_video,
                "message": f"label_video '{label_video}' does not match path (expected '{inferred_label}')"
            })
    
    # 5. Check source_face consistency with path
    if rel_path:
        inferred_source_face = infer_source_face(rel_path, dataset)
        if source_face != inferred_source_face:
            errors.append({
                "row": row_num,
                "type": "source_face_mismatch",
                "field": "source_face",
                "value": source_face,
                "message": f"source_face '{source_face}' does not match path (expected '{inferred_source_face}')"
            })
    
    # 6. Check video_id matches filename stem
    if rel_path and video_id:
        path_obj = Path(rel_path)
        expected_stem = path_obj.stem
        if video_id != expected_stem:
            # For CelebDF-v2, video_id is a hash, so this check doesn't apply
            if dataset == "DF40":
                errors.append({
                    "row": row_num,
                    "type": "video_id_mismatch",
                    "field": "video_id",
                    "value": video_id,
                    "message": f"video_id '{video_id}' does not match filename stem '{expected_stem}'"
                })
    
    # 7. Check ext matches file extension
    if rel_path and ext:
        path_obj = Path(rel_path)
        expected_ext = path_obj.suffix.lower().lstrip(".")
        if ext.lower() != expected_ext:
            errors.append({
                "row": row_num,
                "type": "ext_mismatch",
                "field": "ext",
                "value": ext,
                "message": f"ext '{ext}' does not match file extension '{expected_ext}'"
            })
    
    # 8. Check for missing required fields
    # Note: generator can be empty for CelebDF-v2, so it's not strictly required
    required_fields = ["dataset", "split", "video_id", "rel_path", "label_video"]
    for field in required_fields:
        if not (row.get(field) or "").strip():
            errors.append({
                "row": row_num,
                "type": "missing_field",
                "field": field,
                "value": "",
                "message": f"Required field '{field}' is missing or empty"
            })
    
    # Generator is required for DF40, but can be empty for CelebDF-v2
    if dataset == "DF40" and not (row.get("generator") or "").strip():
        errors.append({
            "row": row_num,
            "type": "missing_field",
            "field": "generator",
            "value": "",
            "message": "Required field 'generator' is missing or empty for DF40 dataset"
        })
    
    return errors


def main():
    ROOT = Path(__file__).resolve().parents[1]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading manifest: {MANIFEST}")
    print(f"Root directory: {ROOT}")
    
    rows = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total rows to validate: {len(rows)}")
    
    all_errors = []
    error_counts = defaultdict(int)
    
    for i, row in enumerate(rows, start=2):  # Start at 2 because row 1 is header
        errors = validate_row(row, i, ROOT)
        all_errors.extend(errors)
        for err in errors:
            error_counts[err["type"]] += 1
        
        if (i - 1) % 1000 == 0:
            print(f"Validated {i - 1} rows...")
    
    # Write detailed error report
    error_file = REPORT_DIR / "errors.json"
    with error_file.open("w", encoding="utf-8") as f:
        json.dump(all_errors, f, indent=2, ensure_ascii=False)
    
    # Write summary
    summary = {
        "total_rows": len(rows),
        "total_errors": len(all_errors),
        "rows_with_errors": len(set(err["row"] for err in all_errors)),
        "error_counts": dict(error_counts),
        "error_file": str(error_file)
    }
    
    summary_file = REPORT_DIR / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total rows validated: {len(rows)}")
    print(f"Total errors found: {len(all_errors)}")
    print(f"Rows with errors: {summary['rows_with_errors']}")
    print(f"\nError breakdown:")
    for err_type, count in sorted(error_counts.items()):
        print(f"  {err_type}: {count}")
    print(f"\nDetailed errors written to: {error_file}")
    print(f"Summary written to: {summary_file}")
    
    if all_errors:
        print("\n⚠️  VALIDATION FAILED - Issues found in manifest")
        # Show first 10 errors as examples
        print("\nFirst 10 errors:")
        for err in all_errors[:10]:
            print(f"  Row {err['row']}: {err['type']} - {err['message']}")
        return 1
    else:
        print("\n✅ VALIDATION PASSED - All rows are valid")
        return 0


if __name__ == "__main__":
    exit(main())

