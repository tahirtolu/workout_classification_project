"""
Eğitim Grafikleri Çizim Scripti

Bu script, training_history.json dosyalarından loss ve accuracy grafikleri çizer.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Windows encoding ayarları
if sys.platform == 'win32':
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# Türkçe font ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_training_history(model_path):
    """Training history JSON dosyasını yükle"""
    history_path = Path(model_path) / 'training_history.json'
    
    if not history_path.exists():
        print(f"❌ Dosya bulunamadı: {history_path}")
        return None
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    return history


def plot_training_curves(history, model_name, output_dir='outputs/training_curves'):
    """
    Loss ve Accuracy grafiklerini çiz
    
    Args:
        history: Training history dict
        model_name: Model adı (örn: 'image_classifier', 'sequence_classifier')
        output_dir: Çıktı klasörü
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Figure oluştur: 2 subplot yan yana
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== SOL GRAFİK: LOSS =====
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])
    
    # Loss grafiği aşağı doğru inmeli (zaten öyle olacak)
    
    # ===== SAĞ GRAFİK: ACCURACY =====
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 105])  # Accuracy 0-100% arası
    
    # Accuracy grafiği yukarı doğru çıkmalı (zaten öyle olacak)
    
    plt.tight_layout()
    
    # Kaydet
    output_file = output_path / f'{model_name}_training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Grafik kaydedildi: {output_file}")
    
    plt.close()
    
    return output_file


def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Eğitim Grafikleri Çizim Scripti')
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['image', 'sequence', 'both'],
        default='both',
        help='Grafik çizilecek model tipi'
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='models',
        help='Modeller klasörü yolu'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/training_curves',
        help='Çıktı klasörü'
    )
    
    args = parser.parse_args()
    
    # Çalışma dizinini proje kök dizinine ayarla
    os.chdir(project_root)
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*60)
    print("EGITIM GRAFIKLERI CIZILIYOR")
    print("="*60)
    
    if args.model_type in ['image', 'both']:
        image_model_path = models_dir / 'image_classifier'
        if image_model_path.exists():
            print(f"\n📊 Image Classifier grafikleri çiziliyor...")
            history = load_training_history(image_model_path)
            if history:
                plot_training_curves(history, 'image_classifier', output_dir)
                print(f"   ✅ Image model grafikleri hazır!")
        else:
            print(f"⚠️  Image model bulunamadı: {image_model_path}")
    
    if args.model_type in ['sequence', 'both']:
        sequence_model_path = models_dir / 'sequence_classifier'
        if sequence_model_path.exists():
            print(f"\n📊 Sequence Classifier grafikleri çiziliyor...")
            history = load_training_history(sequence_model_path)
            if history:
                plot_training_curves(history, 'sequence_classifier', output_dir)
                print(f"   ✅ Sequence model grafikleri hazır!")
        else:
            print(f"⚠️  Sequence model bulunamadı: {sequence_model_path}")
    
    print("\n" + "="*60)
    print("✅ TUM GRAFIKLER HAZIR!")
    print("="*60)
    print(f"\n📁 Çıktı klasörü: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

