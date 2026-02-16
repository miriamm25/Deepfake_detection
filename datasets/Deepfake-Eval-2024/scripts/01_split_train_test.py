#!/usr/bin/env python3
"""
Organize video files by train/test split and create enhanced metadata.
Creates organized/train/ and organized/test/ directories with symlinks or copies.
Also generates a cleaner metadata CSV with boolean columns for easier filtering.
"""

import pandas as pd
import os
import shutil
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    metadata_file = base_dir / "video-metadata-publish-with-links.csv"
    video_data_dir = base_dir / "video-data"
    organized_dir = base_dir / "organized"
    
    # Read metadata
    print(f"Reading metadata from {metadata_file}...")
    df = pd.read_csv(metadata_file)
    print(f"Found {len(df)} videos in metadata")
    
    # Create enhanced metadata with boolean columns
    print("\nCreating enhanced metadata...")
    df_enhanced = df.copy()
    
    # Add local path
    df_enhanced["local_path"] = "video-data/" + df_enhanced["Filename"]
    
    # Add boolean columns for easier filtering
    df_enhanced["is_video_fake"] = df_enhanced["Video Ground Truth"] == "Fake"
    df_enhanced["is_audio_fake"] = df_enhanced["Audio Ground Truth"] == "Fake"
    df_enhanced["is_video_real"] = df_enhanced["Video Ground Truth"] == "Real"
    df_enhanced["is_audio_real"] = df_enhanced["Audio Ground Truth"] == "Real"
    df_enhanced["is_audio_unknown"] = df_enhanced["Audio Ground Truth"] == "Unknown"
    
    # Combined labels
    df_enhanced["any_fake"] = df_enhanced["is_video_fake"] | df_enhanced["is_audio_fake"]
    df_enhanced["both_real"] = df_enhanced["is_video_real"] & df_enhanced["is_audio_real"]
    
    # Create label combinations for easier categorization
    def get_category(row):
        video_label = row["Video Ground Truth"]
        audio_label = row["Audio Ground Truth"]
        if video_label == "Fake" and audio_label == "Fake":
            return "both_fake"
        elif video_label == "Fake" and audio_label == "Real":
            return "video_fake_audio_real"
        elif video_label == "Real" and audio_label == "Fake":
            return "video_real_audio_fake"
        elif video_label == "Real" and audio_label == "Real":
            return "both_real"
        elif video_label == "Fake" and audio_label == "Unknown":
            return "video_fake_audio_unknown"
        elif video_label == "Real" and audio_label == "Unknown":
            return "video_real_audio_unknown"
        else:
            return "unknown"
    
    df_enhanced["category"] = df_enhanced.apply(get_category, axis=1)
    
    # Rename Finetuning Set to split for clarity
    df_enhanced = df_enhanced.rename(columns={"Finetuning Set": "split"})
    
    # Save enhanced metadata
    output_metadata = base_dir / "video-metadata-enhanced.csv"
    df_enhanced.to_csv(output_metadata, index=False)
    print(f"Saved enhanced metadata to {output_metadata}")
    
    # Organize files by train/test split
    print("\nOrganizing files by train/test split...")
    train_dir = organized_dir / "train"
    test_dir = organized_dir / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Count files
    train_count = 0
    test_count = 0
    missing_count = 0
    
    for _, row in df_enhanced.iterrows():
        filename = row["Filename"]
        split = row["split"]
        source_path = video_data_dir / filename
        dest_dir = train_dir if split == "train" else test_dir
        dest_path = dest_dir / filename
        
        if not source_path.exists():
            missing_count += 1
            print(f"  Warning: {filename} not found in video-data/")
            continue
        
        # Create symlink (or copy if symlinks don't work)
        try:
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            dest_path.symlink_to(source_path.absolute())
            if split == "train":
                train_count += 1
            else:
                test_count += 1
        except OSError:
            # If symlink fails (e.g., on Windows or across filesystems), copy instead
            shutil.copy2(source_path, dest_path)
            if split == "train":
                train_count += 1
            else:
                test_count += 1
    
    print(f"\nOrganization complete!")
    print(f"  Train: {train_count} files")
    print(f"  Test: {test_count} files")
    print(f"  Missing: {missing_count} files")
    print(f"\nFiles organized in: {organized_dir}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Dataset Summary:")
    print("="*60)
    print(f"\nSplit distribution:")
    print(df_enhanced["split"].value_counts().to_string())
    
    print(f"\nVideo Ground Truth:")
    print(df_enhanced["Video Ground Truth"].value_counts().to_string())
    
    print(f"\nAudio Ground Truth:")
    print(df_enhanced["Audio Ground Truth"].value_counts().to_string())
    
    print(f"\nCategory distribution:")
    print(df_enhanced["category"].value_counts().to_string())
    
    print(f"\nCategory by split:")
    print(pd.crosstab(df_enhanced["split"], df_enhanced["category"]).to_string())
    
    print("\n" + "="*60)
    print("Done! You can now use:")
    print(f"  - Enhanced metadata: {output_metadata}")
    print(f"  - Organized videos: {organized_dir}/train/ and {organized_dir}/test/")
    print("="*60)

if __name__ == "__main__":
    main()

