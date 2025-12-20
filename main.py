"""
Yapay Zeka Destekli Kişisel Antrenör - Ana Uygulama

Bu script, egzersiz videolarını analiz eder ve form kontrolü yapar.
"""

import os
import sys

# Proje kök dizinini Python path'ine ekle
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Unicode çıktı için encoding ayarla (Windows)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.pose_detector import PoseDetector


def main():
    """Ana uygulama fonksiyonu"""
    print("=" * 60)
    print("Yapay Zeka Destekli Kişisel Antrenör")
    print("=" * 60)
    
    # Çalışma dizinini proje kök dizinine ayarla
    os.chdir(project_root)
    print(f"\nÇalışma dizini: {os.getcwd()}")
    
    # Komut satırı argümanları kontrol et
    auto_mode = False
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--test', '-t', '--auto']:
            auto_mode = True
            if len(sys.argv) > 2:
                video_path = sys.argv[2]
            else:
                print("\n❌ Hata: Test modu için video dosyası yolu gerekli!")
                print("Kullanım: python main.py --test video_dosyasi.mp4")
                sys.exit(1)
        else:
            video_path = sys.argv[1]
    else:
        # Video dosyası kontrolü
        video_path = input("\nVideo dosyası yolunu girin: ").strip()
        if not video_path:
            print("\n❌ Hata: Video dosyası yolu gerekli!")
            print("Örnek: python main.py data/raw_videos/squat/squat_001.mp4")
            sys.exit(1)
    
    if not os.path.exists(video_path):
        print(f"\n❌ Hata: '{video_path}' dosyası bulunamadı!")
        print("Lütfen video dosyasının yolunu kontrol edin.")
        sys.exit(1)
    
    # Çıktı klasörü oluştur
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Çıktı dosya yolları
    output_video = os.path.join(output_dir, "output_video.mp4")
    output_keypoints = os.path.join(output_dir, "keypoints.npy")
    
    # Pose detector oluştur
    print("\n📹 Pose detector başlatılıyor...")
    detector = PoseDetector()
    
    try:
        # Videoyu işle
        print(f"\n🔄 Video işleniyor: {video_path}")
        print("   (Çıkmak için 'q' tuşuna basın)\n")
        
        keypoints = detector.process_video(
            video_path=video_path,
            output_path=output_video,
            display=True  # Görüntüyü göster
        )
        
        # Keypoints'i kaydet
        if keypoints is not None:
            import numpy as np
            np.save(output_keypoints, keypoints)
            print(f"\n✅ Keypoints kaydedildi: {output_keypoints}")
            print(f"   Şekil: {keypoints.shape}")
            print(f"   Her frame için {keypoints.shape[1]} özellik")
        else:
            print("\n⚠️  Uyarı: Hiç pose tespit edilemedi!")
            
        print(f"\n✅ Çıktı videosu kaydedildi: {output_video}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.release()
        print("\n✅ İşlem tamamlandı")


if __name__ == "__main__":
    main()

