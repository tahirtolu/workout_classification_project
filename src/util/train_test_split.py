"""
Train/Test Split Scripti

Bu script, raw_data klasöründeki videoları train/test'e böler.
- Videolar yalnızca raw_data/videos içeriğinden alınır
- Görseller üretilmez; train/videos çıktısı frame üretimi için kaynak olacaktır
"""

import os
import shutil
import random
from pathlib import Path
import sys

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_exercises(raw_data_dir):
    """raw_data/videos/ klasöründeki tüm egzersizleri listeler"""
    videos_dir = Path(raw_data_dir) / "videos"
    if not videos_dir.exists():
        print(f"❌ Hata: {videos_dir} bulunamadı!")
        return []
    
    exercises = [d.name for d in videos_dir.iterdir() if d.is_dir()]
    return sorted(exercises)


def get_video_files(exercise_dir):
    """Bir egzersiz klasöründeki tüm video dosyalarını listeler"""
    video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI']
    videos = []
    for ext in video_extensions:
        videos.extend(list(exercise_dir.glob(f"*{ext}")))
    return sorted(videos)


def split_videos(exercise_dir, train_ratio=0.8, seed=42):
    """
    Videoları train/test'e böler
    
    Args:
        exercise_dir: Egzersiz klasörü (Path)
        train_ratio: Train oranı (0.8 = %80 train, %20 test)
        seed: Random seed (reproducibility için)
    
    Returns:
        (train_videos, test_videos): İki liste
    """
    videos = get_video_files(exercise_dir)
    
    if len(videos) == 0:
        return [], []
    
    # Random seed ayarla (her çalıştırmada aynı split için)
    random.seed(seed)
    
    # Videoları karıştır
    shuffled = videos.copy()
    random.shuffle(shuffled)
    
    # Split index hesapla
    split_idx = int(len(shuffled) * train_ratio)
    
    train_videos = shuffled[:split_idx]
    test_videos = shuffled[split_idx:]
    
    return train_videos, test_videos


def copy_files(files, dest_dir, file_type="video"):
    """Dosyaları hedef klasöre kopyalar"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for src_file in files:
        dest_file = dest_dir / src_file.name
        try:
            shutil.copy2(src_file, dest_file)
            copied += 1
        except Exception as e:
            print(f"  ⚠️  Kopyalama hatası ({src_file.name}): {e}")
    
    return copied


def create_train_test_split(raw_data_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Ana fonksiyon: Train/test split yapar
    
    Args:
        raw_data_dir: raw_data klasörü yolu (örn: "data/raw_data")
        output_dir: Çıktı klasörü (örn: "data")
        train_ratio: Train oranı (0.8 = %70)
        seed: Random seed
    """
    raw_data_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    
    if not raw_data_path.exists():
        print(f"❌ Hata: {raw_data_path} bulunamadı!")
        return
    
    # Klasör yapılarını oluştur
    train_videos_dir = output_path / "train" / "videos"
    test_videos_dir = output_path / "test" / "videos"
    train_images_dir = output_path / "train" / "images"  # yalnızca iskelet, otomatik frame üretimi için
    
    # Temizlik: Önceki split'i temizle (isteğe bağlı)
    if train_videos_dir.exists():
        response = input(f"\n⚠️  {train_videos_dir} zaten var. Üzerine yazılsın mı? (e/h): ")
        if response.lower() != 'e':
            print("İşlem iptal edildi.")
            return
        shutil.rmtree(train_videos_dir.parent, ignore_errors=True)
    
    # Egzersizleri al
    exercises = get_exercises(raw_data_path)
    if not exercises:
        print("❌ Hiç egzersiz bulunamadı!")
        return
    
    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT İŞLEMİ BAŞLIYOR")
    print(f"{'='*60}")
    print(f"\n📊 Toplam egzersiz sayısı: {len(exercises)}")
    print(f"📁 Raw data: {raw_data_path}")
    print(f"📁 Output: {output_path}")
    print(f"📈 Train oranı: {train_ratio*100:.0f}% / Test oranı: {(1-train_ratio)*100:.0f}%")
    print(f"🎲 Random seed: {seed}")
    print(f"\n{'='*60}\n")
    
    # İstatistikler
    stats = {
        'total_videos': 0,
        'train_videos': 0,
        'test_videos': 0,
        'exercises': {}
    }
    
    # Her egzersiz için işlem yap
    for i, exercise in enumerate(exercises, 1):
        print(f"\n[{i}/{len(exercises)}] 🔄 İşleniyor: {exercise}")
        
        # Videoları al
        exercise_videos_dir = raw_data_path / "videos" / exercise
        videos = get_video_files(exercise_videos_dir)
        
        if not videos:
            print(f"  ⚠️  {exercise} için video bulunamadı, atlanıyor...")
            continue
        
        stats['total_videos'] += len(videos)
        
        # Train/test split
        train_videos, test_videos = split_videos(exercise_videos_dir, train_ratio, seed)
        
        print(f"  📹 Toplam: {len(videos)} video")
        print(f"  ✅ Train: {len(train_videos)} video")
        print(f"  ✅ Test: {len(test_videos)} video")
        
        stats['train_videos'] += len(train_videos)
        stats['test_videos'] += len(test_videos)
        
        # Videoları kopyala (sadece video odaklı)
        train_exercise_dir = train_videos_dir / exercise
        test_exercise_dir = test_videos_dir / exercise
        
        train_copied = copy_files(train_videos, train_exercise_dir, "video")
        test_copied = copy_files(test_videos, test_exercise_dir, "video")
        
        print(f"  📋 Train videoları kopyalandı: {train_copied}/{len(train_videos)}")
        print(f"  📋 Test videoları kopyalandı: {test_copied}/{len(test_videos)}")
        
        # İstatistikleri kaydet
        stats['exercises'][exercise] = {
            'total_videos': len(videos),
            'train_videos': len(train_videos),
            'test_videos': len(test_videos)
        }
    
    # Özet
    print(f"\n{'='*60}")
    print("✅ İŞLEM TAMAMLANDI")
    print(f"{'='*60}")
    print(f"\n📊 ÖZET İSTATİSTİKLER:")
    print(f"  📹 Toplam video: {stats['total_videos']}")
    print(f"    ✅ Train: {stats['train_videos']} ({stats['train_videos']/stats['total_videos']*100:.1f}%)")
    print(f"    ✅ Test: {stats['test_videos']} ({stats['test_videos']/stats['total_videos']*100:.1f}%)")
    print(f"\n📁 Çıktı klasörleri:")
    print(f"  ✅ Train videoları: {train_videos_dir}")
    print(f"  ✅ Test videoları: {test_videos_dir}")
    print(f"\n{'='*60}\n")


def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train/Test Split Scripti')
    parser.add_argument(
        '--raw_data',
        type=str,
        default='data/raw_data',
        help='Raw data klasörü yolu (varsayılan: data/raw_data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Çıktı klasörü yolu (varsayılan: data)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Train oranı (varsayılan: 0.8 = %80)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (varsayılan: 42)'
    )
    
    args = parser.parse_args()
    
    # Çalışma dizinini proje kök dizinine ayarla
    os.chdir(project_root)
    
    create_train_test_split(
        raw_data_dir=args.raw_data,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

