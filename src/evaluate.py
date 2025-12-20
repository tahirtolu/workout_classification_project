"""
Model Değerlendirme Scripti

Bu script, eğitilmiş modelleri test verisi üzerinde değerlendirir:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Sınıf bazlı metrikler
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import ExerciseImageDataset, ExerciseVideoDataset
from src.models.exercise_classifier import ImageClassifier, SequenceClassifier
from torch.utils.data import DataLoader

# Windows encoding ayarları
if sys.platform == 'win32':
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass


def evaluate_image_model(
    model_path,
    data_dir='data/processed',
    batch_size=32,
    device=None,
    output_dir='outputs/evaluation'
):
    """Görsel modeli değerlendir"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("GORSEL MODEL DEGERLENDIRME")
    print("="*60)
    
    # Model yükle
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Sınıf isimlerini yükle
    model_dir = Path(model_path).parent
    class_names_path = model_dir / 'class_names.json'
    
    if class_names_path.exists():
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        num_classes = len(class_names)
    else:
        # Eğer class_names.json yoksa, checkpoint'ten al
        num_classes = checkpoint.get('num_classes', 22)
        class_names = [f'Class_{i}' for i in range(num_classes)]
        print(f"[UYARI] class_names.json bulunamadi, varsayilan isimler kullaniliyor")
    
    # Model oluştur ve yükle
    model = ImageClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model yuklendi: {model_path}")
    print(f"Sinif sayisi: {num_classes}")
    print(f"Device: {device}")
    
    # Test veri setini yükle
    # Not: Test veri setinde görseller yok, bu yüzden validation setini kullanıyoruz
    print("\nTest veri seti yukleniyor...")
    print("[NOT] Test veri setinde gorseller yok, validation seti kullaniliyor...")
    
    # Validation setini test olarak kullan
    from src.data_loader import create_data_loaders
    _, val_loader_images, _, _, _, _ = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=0.2,
        num_workers=0
    )
    
    # Validation dataset'ini al (Subset'ten ana dataset'e eriş)
    test_dataset = val_loader_images.dataset
    test_class_names = class_names  # Model'deki sınıf isimlerini kullan
    
    # Validation loader'ı direkt kullan (zaten DataLoader)
    test_loader = val_loader_images
    
    # Tahminler
    print("\nTahminler yapiliyor...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrikleri hesapla
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("DEGERLENDIRME SONUCLARI (Validation Set)")
    print("="*60)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print(f"Toplam ornek: {len(all_labels)}")
    
    # Confusion Matrix görselleştir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Ornek Sayisi'}
    )
    plt.title('Gorsel Model - Confusion Matrix (Validation Set)')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gercek')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path / 'image_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix kaydedildi: {output_path / 'image_model_confusion_matrix.png'}")
    plt.close()
    
    # Sınıf bazlı rapor
    report = classification_report(
        all_labels, all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # Raporu kaydet
    report_path = output_path / 'image_model_classification_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Sinif bazli rapor kaydedildi: {report_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def evaluate_sequence_model(
    model_path,
    data_dir='data/processed',
    batch_size=16,
    sequence_length=60,
    device=None,
    output_dir='outputs/evaluation'
):
    """Video modeli değerlendir"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("VIDEO MODEL DEGERLENDIRME")
    print("="*60)
    
    # Model yükle
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Sınıf isimlerini yükle
    model_dir = Path(model_path).parent
    class_names_path = model_dir / 'class_names.json'
    
    if class_names_path.exists():
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        num_classes = len(class_names)
    else:
        num_classes = checkpoint.get('num_classes', 22)
        class_names = [f'Class_{i}' for i in range(num_classes)]
        print(f"[UYARI] class_names.json bulunamadi, varsayilan isimler kullaniliyor")
    
    # Model oluştur ve yükle
    model = SequenceClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model yuklendi: {model_path}")
    print(f"Sinif sayisi: {num_classes}")
    print(f"Device: {device}")
    
    # Test veri setini yükle
    print("\nTest veri seti yukleniyor...")
    test_dataset = ExerciseVideoDataset(
        data_dir, split='test', sequence_length=sequence_length
    )
    
    # DataLoader oluştur
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Tahminler
    print("\nTahminler yapiliyor...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrikleri hesapla
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("DEGERLENDIRME SONUCLARI")
    print("="*60)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print(f"Toplam ornek: {len(all_labels)}")
    
    # Confusion Matrix görselleştir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Ornek Sayisi'}
    )
    plt.title('Video Model - Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gercek')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path / 'sequence_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix kaydedildi: {output_path / 'sequence_model_confusion_matrix.png'}")
    plt.close()
    
    # Sınıf bazlı rapor
    report = classification_report(
        all_labels, all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # Raporu kaydet
    report_path = output_path / 'sequence_model_classification_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Sinif bazli rapor kaydedildi: {report_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Model Değerlendirme Scripti')
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['image', 'sequence', 'both'],
        default='both',
        help='Değerlendirilecek model tipi'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Model dosya yolu (opsiyonel, en iyi model otomatik bulunur)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Veri klasörü yolu'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch boyutu'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=60,
        help='Video sequence uzunluğu'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/evaluation',
        help='Çıktı klasörü'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu), None ise otomatik'
    )
    
    args = parser.parse_args()
    
    # Device belirle
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Çalışma dizinini proje kök dizinine ayarla
    os.chdir(project_root)
    
    # Model yolu belirle
    if args.model_path is None:
        # En iyi modeli bul
        if args.model_type in ['image', 'both']:
            image_model_dir = Path('models/image_classifier')
            if image_model_dir.exists():
                # En son best model'i bul
                best_models = sorted(image_model_dir.glob('best_model_epoch_*.pth'))
                if best_models:
                    image_model_path = best_models[-1]
                else:
                    image_model_path = image_model_dir / 'final_model.pth'
            else:
                print("[HATA] Gorsel model bulunamadi!")
                image_model_path = None
        else:
            image_model_path = None
        
        if args.model_type in ['sequence', 'both']:
            sequence_model_dir = Path('models/sequence_classifier')
            if sequence_model_dir.exists():
                best_models = sorted(sequence_model_dir.glob('best_model_epoch_*.pth'))
                if best_models:
                    sequence_model_path = best_models[-1]
                else:
                    sequence_model_path = sequence_model_dir / 'final_model.pth'
            else:
                print("[HATA] Video model bulunamadi!")
                sequence_model_path = None
        else:
            sequence_model_path = None
    else:
        # Kullanıcı model yolu verdi
        if args.model_type == 'image':
            image_model_path = Path(args.model_path)
            sequence_model_path = None
        elif args.model_type == 'sequence':
            image_model_path = None
            sequence_model_path = Path(args.model_path)
        else:
            print("[HATA] model_type='both' ile --model_path kullanilamaz!")
            return
    
    # Değerlendirme yap
    results = {}
    
    if args.model_type in ['image', 'both'] and image_model_path and image_model_path.exists():
        results['image'] = evaluate_image_model(
            model_path=str(image_model_path),
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir
        )
    
    if args.model_type in ['sequence', 'both'] and sequence_model_path and sequence_model_path.exists():
        results['sequence'] = evaluate_sequence_model(
            model_path=str(sequence_model_path),
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            device=device,
            output_dir=args.output_dir
        )
    
    # Özet
    print("\n" + "="*60)
    print("OZET")
    print("="*60)
    for model_type, result in results.items():
        print(f"\n{model_type.upper()} Model:")
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  F1-Score: {result['f1']*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

