"""
Veri Toplama ve Keypoints Çıkarma Modülü

Bu modül, train/test split edilmiş görseller ve videolardan keypoints çıkarır.
- Görsellerden keypoints çıkar (statik pozlar)
- Videolardan keypoints çıkar (zaman serisi)
- Çıktıları organize eder ve kaydeder
"""

import os
import sys
from pathlib import Path
import numpy as np

# tqdm import
from tqdm import tqdm

# Windows'ta stdout/stderr encoding ayarları
if sys.platform == 'win32':
    try:
        import io
        # UTF-8 encoding ayarla
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import'u düzelt
try:
    from src.pose_detector import PoseDetector
except ImportError:
    # Alternatif import yolu
    from pose_detector import PoseDetector

import cv2

class DataCollector:
    """Veri toplama ve keypoints çıkarma sınıfı"""
    
    def __init__(self):
        """Pose detector'ı başlat ve GPU durumunu kontrol et"""
        # GPU durumunu kontrol et ve göster
        self._check_and_show_gpu()
        
        print("📹 Pose detector başlatılıyor...")
        self.detector = PoseDetector()
        print("✅ Pose detector hazır!\n")
    
    def _check_and_show_gpu(self):
        """GPU durumunu kontrol eder ve bilgi gösterir"""
        print("\n" + "="*60)
        print("GPU DURUMU KONTROL EDİLİYOR")
        print("="*60)
        
        try:
            import torch
        except ImportError:
            print("\n⚠️  PyTorch yüklü değil - GPU kontrolü yapılamadı")
            print("   ℹ️  MediaPipe CPU kullanmaya devam edecek")
            print("="*60 + "\n")
            return

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\n✅ GPU bulundu: {gpu_count} adet")

            for idx in range(gpu_count):
                props = torch.cuda.get_device_properties(idx)
                total_mem_gb = props.total_memory / (1024 ** 3)
                capability = ".".join(map(str, torch.cuda.get_device_capability(idx)))

                print(f"\n   GPU {idx+1}:")
                print(f"      Ad: {props.name}")
                print(f"      Bellek: {total_mem_gb:.2f} GB")
                print(f"      Compute Capability: {capability}")

            cuda_version = torch.version.cuda or "Bilinmiyor"
            print(f"\n   📊 PyTorch CUDA versiyonu: {cuda_version}")
            print(f"   ℹ️  MediaPipe pose detection CPU kullanır")
            print(f"   ℹ️  GPU avantajı PyTorch modelleri eğitilirken kullanılacak")
        else:
            print("\n⚠️  GPU bulunamadı - CPU modunda çalışacak")
            print("   ℹ️  Keypoints çıkarma CPU'da yapılacak (MediaPipe)")
            print("   ℹ️  PyTorch modelleri de CPU'da eğitilecek")
        
        print("="*60 + "\n")
    
    def process_image(self, image_path):
        """
        Tek bir görselden keypoints çıkarır
        
        Args:
            image_path: Görsel dosya yolu (Path veya str)
        
        Returns:
            keypoints: (132,) numpy array veya None
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return None
        
        # Görseli oku
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Pose detection
        results = self.detector.process_frame(image)
        
        # Keypoints çıkar
        keypoints = self.detector.extract_keypoints(results)
        
        return keypoints
    
    def process_images_folder(self, images_dir, output_dir, exercise_name=None):
        """
        Bir klasördeki tüm görsellerden keypoints çıkarır
        
        Args:
            images_dir: Görsel klasörü (Path)
            output_dir: Çıktı klasörü (Path)
            exercise_name: Egzersiz adı (opsiyonel, klasör adından otomatik)
        
        Returns:
            keypoints_array: (num_images, 132) numpy array
            stats: İstatistikler dict
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not images_dir.exists():
            print(f"⚠️  Klasör bulunamadı: {images_dir}")
            return None, None
        
        # Egzersiz adını belirle
        if exercise_name is None:
            exercise_name = images_dir.name
        
        # Tüm görselleri al
        image_files = sorted(list(images_dir.glob("*.jpg")))
        
        if not image_files:
            print(f"⚠️  {images_dir} klasöründe görsel bulunamadı")
            return None, None
        
        print(f"\n🖼️  İşleniyor: {exercise_name}")
        print(f"   📁 Klasör: {images_dir}")
        print(f"   📊 Toplam görsel: {len(image_files)}")
        
        all_keypoints = []
        processed = 0
        failed = 0
        
        # Her görseli işle (tqdm progress bar ile)
        try:
            # tqdm progress bar kullan
            pbar = tqdm(
                image_files, 
                desc=f"  İşleniyor ({exercise_name})",
                unit="görsel",
                ncols=100,  # Progress bar genişliği
                file=sys.stdout,
                disable=False
            )
            
            for image_file in pbar:
                try:
                    keypoints = self.process_image(image_file)
                    
                    if keypoints is not None:
                        all_keypoints.append(keypoints)
                        processed += 1
                    else:
                        failed += 1
                    
                    # Progress bar'ı güncelle
                    pbar.set_postfix({
                        'Başarılı': processed,
                        'Başarısız': failed,
                        'İlerleme': f"{processed}/{len(image_files)}"
                    })
                except Exception as e:
                    failed += 1
                    if failed <= 5:  # İlk 5 hatayı göster
                        pbar.write(f"   ⚠️  Hata ({image_file.name}): {e}")
            
            pbar.close()
            
        except Exception as e:
            print(f"   ❌ Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()
        
        # NumPy array'e dönüştür
        if all_keypoints:
            keypoints_array = np.array(all_keypoints)
            
            # Kaydet
            output_file = output_dir / f"{exercise_name}_keypoints.npy"
            np.save(output_file, keypoints_array)
            
            print(f"   ✅ İşlenen: {processed}/{len(image_files)}")
            print(f"   ❌ Başarısız: {failed}/{len(image_files)}")
            print(f"   💾 Kaydedildi: {output_file}")
            print(f"   📊 Şekil: {keypoints_array.shape}")
            
            stats = {
                'exercise': exercise_name,
                'total_images': len(image_files),
                'processed': processed,
                'failed': failed,
                'success_rate': processed / len(image_files) * 100,
                'output_file': str(output_file),
                'shape': keypoints_array.shape
            }
            
            return keypoints_array, stats
        else:
            print(f"   ❌ Hiç keypoints çıkarılamadı!")
            return None, None
    
    def process_video(self, video_path):
        """
        Tek bir videodan keypoints çıkarır
        
        Args:
            video_path: Video dosya yolu (Path veya str)
        
        Returns:
            keypoints: (frame_count, 132) numpy array veya None
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            return None
        
        # Video'yu işle (display=False, output_path=None, verbose=False - sessiz mod)
        keypoints = self.detector.process_video(
            video_path=str(video_path),
            output_path=None,
            display=False,
            verbose=False  # Sessiz mod - print mesajları gösterme
        )
        
        return keypoints
    
    def process_videos_folder(self, videos_dir, output_dir, exercise_name=None):
        """
        Bir klasördeki tüm videolardan keypoints çıkarır
        
        Args:
            videos_dir: Video klasörü (Path)
            output_dir: Çıktı klasörü (Path)
            exercise_name: Egzersiz adı (opsiyonel, klasör adından otomatik)
        
        Returns:
            stats: İstatistikler dict
        """
        videos_dir = Path(videos_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not videos_dir.exists():
            print(f"⚠️  Klasör bulunamadı: {videos_dir}")
            return None
        
        # Egzersiz adını belirle
        if exercise_name is None:
            exercise_name = videos_dir.name
        
        # Tüm videoları al
        video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(videos_dir.glob(f"*{ext}")))
        video_files = sorted(video_files)
        
        if not video_files:
            print(f"⚠️  {videos_dir} klasöründe video bulunamadı")
            return None
        
        print(f"\n🎬 İşleniyor: {exercise_name}")
        print(f"   📁 Klasör: {videos_dir}")
        print(f"   📊 Toplam video: {len(video_files)}")
        
        processed = 0
        failed = 0
        total_frames = 0
        
        # Her videoyu işle (tqdm progress bar ile)
        try:
            # tqdm progress bar kullan
            pbar = tqdm(
                video_files,
                desc=f"  İşleniyor ({exercise_name})",
                unit="video",
                ncols=100,  # Progress bar genişliği
                file=sys.stdout,
                disable=False
            )
            
            for video_file in pbar:
                try:
                    keypoints = self.process_video(video_file)
                    
                    if keypoints is not None and len(keypoints) > 0:
                        # Kaydet
                        output_file = output_dir / f"{video_file.stem}_keypoints.npy"
                        np.save(output_file, keypoints)
                        
                        processed += 1
                        total_frames += len(keypoints)
                        
                        # Progress bar'ı güncelle
                        pbar.set_postfix({
                            'Başarılı': processed,
                            'Başarısız': failed,
                            'Frame': len(keypoints),
                            'Toplam Frame': total_frames
                        })
                    else:
                        failed += 1
                        pbar.set_postfix({
                            'Başarılı': processed,
                            'Başarısız': failed
                        })
                except Exception as e:
                    failed += 1
                    pbar.write(f"   ❌ Hata ({video_file.name}): {e}")
                    pbar.set_postfix({
                        'Başarılı': processed,
                        'Başarısız': failed
                    })
            
            pbar.close()
            
        except Exception as e:
            print(f"   ❌ Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"   ✅ İşlenen: {processed}/{len(video_files)}")
        print(f"   ❌ Başarısız: {failed}/{len(video_files)}")
        print(f"   📊 Toplam frame: {total_frames}")
        print(f"   📁 Çıktı klasörü: {output_dir}")
        
        stats = {
            'exercise': exercise_name,
            'total_videos': len(video_files),
            'processed': processed,
            'failed': failed,
            'success_rate': processed / len(video_files) * 100 if video_files else 0,
            'total_frames': total_frames,
            'output_dir': str(output_dir)
        }
        
        return stats
    
    def process_train_data(self, data_dir, output_dir):
        """
        Train verilerini işler (görseller + videolar)
        
        Args:
            data_dir: Train klasörü (data/train)
            output_dir: Çıktı klasörü (data/processed/train)
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        
        print(f"\n{'='*60}")
        print("TRAIN VERİLERİ İŞLENİYOR")
        print(f"{'='*60}")
        
        # Görselleri işle
        images_dir = data_path / "images"
        videos_dir = data_path / "videos"
        
        if images_dir.exists():
            print(f"\n📸 Görseller işleniyor...")
            images_output = output_path / "images"
            
            # Her egzersiz için
            exercise_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
            
            for exercise_dir in exercise_dirs:
                self.process_images_folder(exercise_dir, images_output, exercise_dir.name)
        
        if videos_dir.exists():
            print(f"\n🎬 Videolar işleniyor...")
            videos_output = output_path / "videos"
            
            # Her egzersiz için
            exercise_dirs = sorted([d for d in videos_dir.iterdir() if d.is_dir()])
            
            for exercise_dir in exercise_dirs:
                self.process_videos_folder(exercise_dir, videos_output, exercise_dir.name)
        
        print(f"\n✅ Train verileri işlendi!")
    
    def process_test_data(self, data_dir, output_dir):
        """
        Test verilerini işler (sadece videolar)
        
        Args:
            data_dir: Test klasörü (data/test)
            output_dir: Çıktı klasörü (data/processed/test)
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        
        print(f"\n{'='*60}")
        print("TEST VERİLERİ İŞLENİYOR")
        print(f"{'='*60}")
        
        videos_dir = data_path / "videos"
        
        if videos_dir.exists():
            print(f"\n🎬 Videolar işleniyor...")
            videos_output = output_path / "videos"
            
            # Her egzersiz için
            exercise_dirs = sorted([d for d in videos_dir.iterdir() if d.is_dir()])
            
            for exercise_dir in exercise_dirs:
                self.process_videos_folder(exercise_dir, videos_output, exercise_dir.name)
        
        print(f"\n✅ Test verileri işlendi!")
    
    def release(self):
        """Kaynakları serbest bırak"""
        if self.detector:
            self.detector.release()


def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Veri Toplama ve Keypoints Çıkarma')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Data klasörü yolu (varsayılan: data)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Çıktı klasörü yolu (varsayılan: data/processed)'
    )
    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Sadece train verilerini işle'
    )
    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Sadece test verilerini işle'
    )
    parser.add_argument(
        '--exercise',
        type=str,
        default=None,
        help='Sadece belirtilen egzersizi işle (opsiyonel)'
    )
    
    args = parser.parse_args()
    
    # Çalışma dizinini proje kök dizinine ayarla
    os.chdir(project_root)
    
    # Data collector oluştur
    collector = DataCollector()
    
    try:
        data_path = Path(args.data_dir)
        output_path = Path(args.output_dir)
        
        # Train verilerini işle
        if not args.test_only:
            train_dir = data_path / "train"
            if train_dir.exists():
                train_output = output_path / "train"
                collector.process_train_data(train_dir, train_output)
        
        # Test verilerini işle
        if not args.train_only:
            test_dir = data_path / "test"
            if test_dir.exists():
                test_output = output_path / "test"
                collector.process_test_data(test_dir, test_output)
        
        print(f"\n{'='*60}")
        print("✅ TÜM İŞLEMLER TAMAMLANDI!")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.release()


if __name__ == "__main__":
    main()

