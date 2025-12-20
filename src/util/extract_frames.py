"""
Video Frame Çıkarma Scripti

Bu script, belirli bir klasördeki videolardan kare çıkararak
data/<split>/images yapısını yeniden oluşturur. Amaç:

- Train/test split sonrası sadece ilgili videolardan kare üretmek
- Hedef FPS'e göre frame step belirlemek
- Maksimum kare sayısını kontrol altında tutmak

Örnek kullanım:
    python src/util/extract_frames.py \
        --video_root data/train/videos \
        --image_root data/train/images \
        --target_fps 6 \
        --max_frames 400
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}
DEFAULT_TARGET_FPS = 6.0
DEFAULT_MAX_FRAMES = 400


def list_video_files(video_root: Path):
    """Egzersiz klasörlerini ve içlerindeki videoları döndürür."""
    if not video_root.exists():
        raise FileNotFoundError(f"Video kök klasörü bulunamadı: {video_root}")

    exercises = []
    for exercise_dir in sorted(video_root.iterdir()):
        if exercise_dir.is_dir():
            videos = sorted(
                f for f in exercise_dir.iterdir()
                if f.suffix in VIDEO_EXTENSIONS and f.is_file()
            )
            if videos:
                exercises.append((exercise_dir.name, videos))
    return exercises


def compute_frame_step(original_fps: float, target_fps: float) -> int:
    """Hedef FPS'e göre frame step değeri hesapla."""
    if original_fps <= 0 or target_fps <= 0:
        return 1
    step = max(1, round(original_fps / target_fps))
    return step


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    max_frames: int,
):
    """Tek bir videodan kare çıkar ve kaydet."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "saved": 0,
            "skipped": 0,
            "fps": 0,
            "frame_step": 0,
            "reason": "Video açılamadı",
        }

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = compute_frame_step(original_fps or 30, target_fps)

    saved_frames = 0
    processed_frames = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if processed_frames % frame_step == 0:
            if max_frames and saved_frames >= max_frames:
                break

            frame_name = f"{video_path.stem}_frame_{processed_frames:05d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_frames += 1

        processed_frames += 1

    cap.release()

    return {
        "saved": saved_frames,
        "skipped": max(0, processed_frames - saved_frames),
        "fps": original_fps,
        "frame_step": frame_step,
        "total_frames": total_frames,
    }


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Video frame çıkarma scripti")
    parser.add_argument(
        "--video_root",
        type=Path,
        default=Path("data/train/videos"),
        help="Kare çıkarılacak video klasörü (varsayılan: data/train/videos)",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=Path("data/train/images"),
        help="Karelerin kaydedileceği klasör (varsayılan: data/train/images)",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help=f"Hedef FPS (varsayılan: {DEFAULT_TARGET_FPS})",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Video başına maksimum kare (varsayılan: {DEFAULT_MAX_FRAMES})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Çıktı klasörünü tamamen temizleyip yeniden oluştur",
    )
    return parser


def maybe_clear_output(image_root: Path, overwrite: bool):
    """Overwrite seçiliyse çıktı klasörünü temizle."""
    if overwrite and image_root.exists():
        import shutil

        print(f"⚠️  {image_root} temizleniyor (overwrite seçildi)")
        shutil.rmtree(image_root)
    image_root.mkdir(parents=True, exist_ok=True)


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    video_root = args.video_root
    image_root = args.image_root

    maybe_clear_output(image_root, args.overwrite)

    exercises = list_video_files(video_root)
    if not exercises:
        print(f"❌ Video bulunamadı: {video_root}")
        return

    print("=" * 70)
    print("VIDEO -> FRAME ÇIKARMA İŞLEMİ")
    print("=" * 70)
    print(f"Video kökü   : {video_root}")
    print(f"Görsel kökü  : {image_root}")
    print(f"Hedef FPS    : {args.target_fps}")
    print(f"Maks. kare   : {args.max_frames or 'Sınırsız'}")
    print("=" * 70)

    total_saved = 0
    total_videos = 0

    for exercise_name, videos in exercises:
        print(f"\n🔄 Egzersiz işleniyor: {exercise_name} ({len(videos)} video)")
        exercise_output = image_root / exercise_name
        exercise_output.mkdir(parents=True, exist_ok=True)

        for video_path in tqdm(videos, desc=f"{exercise_name}", unit="video"):
            stats = extract_frames_from_video(
                video_path,
                exercise_output,
                args.target_fps,
                args.max_frames,
            )

            total_saved += stats["saved"]
            total_videos += 1

    print("\n" + "=" * 70)
    print("✅ Kare çıkarma işlemi tamamlandı")
    print(f"Toplam video  : {total_videos}")
    print(f"Toplam kare   : {total_saved}")
    print(f"Çıktı klasörü : {image_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()

