#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, (p.stdout or ""), (p.stderr or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Calea către folderul rădăcină DF40")
    ap.add_argument("--fps", type=int, default=25, help="FPS pentru video-ul rezultat")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    # Ne concentrăm doar pe test/heygen/real
    split_dir = root / "video-data" / "test" / "heygen" / "real"
    out_dir = root / "video-data" / "test" / "heygen" / "real_mp4"

    if not split_dir.exists():
        print(f"[!] Eroare: Nu am găsit folderul sursă: {split_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Listăm toate folderele cu hash-uri
    clip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"[*] Am găsit {len(clip_dirs)} clipuri de procesat în {split_dir.name}")

    for clip_dir in clip_dirs:
        # Numele clipului rămâne cel original (ex: Clip+_HeblzK...)
        output_file = out_dir / f"{clip_dir.name}.mp4"
        
        if output_file.exists():
            print(f"[-] Skipped: {output_file.name} (există deja)")
            continue

        # FFmpeg: transformă secvența %08d.png în mp4
        # Folosim -vf pad pentru a ne asigura că rezoluția e divizibilă cu 2
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-framerate", str(args.fps),
            "-start_number", "0",
            "-i", str(clip_dir / "%08d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output_file)
        ]

        print(f"[>] Se procesează: {clip_dir.name}...", end="\r")
        rc, _, err = run(cmd)
        
        if rc == 0:
            print(f"[OK] Creat: {output_file.name}            ")
        else:
            print(f"[FAIL] Eroare la clipul {clip_dir.name}: {err}")

    print("\n[✓] Procesare finalizată pentru setul de TEST.")

if __name__ == "__main__":
    main()