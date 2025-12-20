"""
Model Eğitim Scripti

Bu script, egzersiz sınıflandırıcı modellerini eğitir:
- ImageClassifier: Görseller için MLP modeli
- SequenceClassifier: Videolar için LSTM modeli
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import create_data_loaders
from src.models.exercise_classifier import ImageClassifier, SequenceClassifier

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


class Trainer:
    """Model eğitimi için Trainer sınıfı"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=0.001,
        num_epochs=50,
        save_dir='models'
    ):
        """
        Args:
            model: PyTorch modeli
            train_loader: Train DataLoader
            val_loader: Validation DataLoader
            device: CUDA veya CPU
            learning_rate: Öğrenme oranı
            num_epochs: Epoch sayısı
            save_dir: Model kayıt klasörü
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss ve optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Eğitim geçmişi
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # En iyi model
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self):
        """Bir epoch eğitim"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Train', ncols=100, file=sys.stdout)
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # İstatistikler
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress bar güncelle
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validation"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Val  ', ncols=100, file=sys.stdout)
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # İstatistikler
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Progress bar güncelle
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Tam eğitim döngüsü"""
        print("\n" + "="*60)
        print("MODEL EGITIMI BASLIYOR")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epoch sayisi: {self.num_epochs}")
        print(f"Model parametre sayisi: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60 + "\n")
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Geçmişi kaydet
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # En iyi modeli kaydet
            if val_acc > self.best_val_acc:
                # Eski best model dosyasını sil (varsa)
                if self.best_model_path and self.best_model_path.exists():
                    try:
                        self.best_model_path.unlink()
                    except Exception as e:
                        print(f"[UYARI] Eski model silinemedi: {e}")
                
                self.best_val_acc = val_acc
                self.best_model_path = self.save_dir / 'best_model.pth'  # Sabit isim
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, self.best_model_path)
                print(f"\n[YENI EN IYI MODEL] Validation Accuracy: {val_acc:.2f}% (Epoch {epoch})")
            
            # Özet
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        
        # Son modeli kaydet
        final_model_path = self.save_dir / 'final_model.pth'
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, final_model_path)
        
        # Geçmişi kaydet
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "="*60)
        print("EGITIM TAMAMLANDI")
        print("="*60)
        print(f"En iyi validation accuracy: {self.best_val_acc:.2f}%")
        print(f"En iyi model: {self.best_model_path}")
        print(f"Final model: {final_model_path}")
        print(f"Geçmiş: {history_path}")
        print("="*60)
        
        return self.history


def train_image_model(
    data_dir='data/processed',
    batch_size=32,
    learning_rate=0.001,
    num_epochs=50,
    val_ratio=0.2,
    save_dir='models/image_classifier',
    device=None
):
    """Görsel modeli eğit"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("GORSEL MODEL EGITIMI")
    print("="*60)
    
    # Veri yükle
    loaders = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=0
    )
    train_loader_images, val_loader_images, _, _, num_classes, class_names = loaders
    
    # Model oluştur
    model = ImageClassifier(num_classes=num_classes)
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        train_loader=train_loader_images,
        val_loader=val_loader_images,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    # Sınıf isimlerini kaydet
    class_names_path = Path(save_dir) / 'class_names.json'
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(class_names.tolist(), f, indent=2, ensure_ascii=False)
    
    # Eğit
    history = trainer.train()
    
    return trainer, history


def train_sequence_model(
    data_dir='data/processed',
    batch_size=16,
    learning_rate=0.001,
    num_epochs=50,
    val_ratio=0.2,
    sequence_length=60,
    save_dir='models/sequence_classifier',
    device=None
):
    """Video modeli eğit"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("VIDEO MODEL EGITIMI")
    print("="*60)
    
    # Veri yükle
    loaders = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        sequence_length=sequence_length,
        num_workers=0
    )
    _, _, train_loader_videos, val_loader_videos, num_classes, class_names = loaders
    
    # Model oluştur
    model = SequenceClassifier(num_classes=num_classes)
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        train_loader=train_loader_videos,
        val_loader=val_loader_videos,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    # Sınıf isimlerini kaydet
    class_names_path = Path(save_dir) / 'class_names.json'
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(class_names.tolist(), f, indent=2, ensure_ascii=False)
    
    # Eğit
    history = trainer.train()
    
    return trainer, history


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Model Eğitim Scripti')
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['image', 'sequence', 'both'],
        default='both',
        help='Eğitilecek model tipi (image, sequence, both)'
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
        '--learning_rate',
        type=float,
        default=0.001,
        help='Öğrenme oranı'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Epoch sayısı'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Validation oranı'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=60,
        help='Video sequence uzunluğu'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models',
        help='Model kayıt klasörü'
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
    
    print("\n" + "="*60)
    print("MODEL EGITIM AYARLARI")
    print("="*60)
    print(f"Model tipi: {args.model_type}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epoch sayisi: {args.num_epochs}")
    print(f"Validation orani: {args.val_ratio}")
    print("="*60)
    
    # Çalışma dizinini proje kök dizinine ayarla
    os.chdir(project_root)
    
    # Model eğit
    if args.model_type in ['image', 'both']:
        train_image_model(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            val_ratio=args.val_ratio,
            save_dir=Path(args.save_dir) / 'image_classifier',
            device=device
        )
    
    if args.model_type in ['sequence', 'both']:
        train_sequence_model(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            val_ratio=args.val_ratio,
            sequence_length=args.sequence_length,
            save_dir=Path(args.save_dir) / 'sequence_classifier',
            device=device
        )
    
    print("\n[OK] Tum egitimler tamamlandi!")


if __name__ == "__main__":
    main()

