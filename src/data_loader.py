"""
Veri Yükleme Modülü

Bu modül, işlenmiş keypoint dosyalarını (.npy) yükler ve
PyTorch DataLoader'a hazır hale getirir.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def normalize_exercise_name(name):
    """
    Egzersiz isimlerini normalize et (büyük/küçük harf, kısaltmalar, vb.)
    
    Args:
        name: Ham egzersiz ismi
    
    Returns:
        normalize edilmiş isim
    """
    name = name.lower().strip()
    
    # Kısaltmalar ve eşleştirmeler (görsel veri setindeki isimlerle eşleştir)
    mappings = {
        'dbp': 'decline bench press',
        'pull up': 'pull Up',
        'pullup': 'pull Up',
        'tricep pushdown': 'tricep Pushdown',
        'tricep push down': 'tricep Pushdown',
        'barbell_hip_thrust_tutorial__gluteworkout': 'hip thrust',
        'hip thrust tutorial': 'hip thrust',
        'barbell hip thrust': 'hip thrust',
        'barbell biceps curl': 'barbell biceps curl',
        'bench press': 'bench press',
        'chest fly machine': 'chest fly machine',
        'deadlift': 'deadlift',
        'decline bench press': 'decline bench press',
        'hammer curl': 'hammer curl',
        'hip thrust': 'hip thrust',
        'incline bench press': 'incline bench press',
        'lat pulldown': 'lat pulldown',
        'lateral raise': 'lateral raise',
        'leg extension': 'leg extension',
        'leg raises': 'leg raises',
        'plank': 'plank',
        'pull Up': 'pull Up',
        'push-up': 'push-up',
        'romanian deadlift': 'romanian deadlift',
        'russian twist': 'russian twist',
        'shoulder press': 'shoulder press',
        'squat': 'squat',
        't bar row': 't bar row',
        'tricep dips': 'tricep dips',
        'tricep Pushdown': 'tricep Pushdown',
    }
    
    # Eşleştirme kontrolü
    if name in mappings:
        return mappings[name]
    
    # Eğer tam eşleşme yoksa, kısmi eşleşme dene
    for key, value in mappings.items():
        if key in name or name in key:
            return value
    
    # Genel normalizasyon: İlk harf büyük, geri kalan küçük
    words = name.split()
    normalized_words = [word.capitalize() for word in words]
    normalized = ' '.join(normalized_words)
    
    return normalized


class ExerciseImageDataset(Dataset):
    """
    Görsellerden çıkarılmış keypoint'ler için Dataset
    
    Her egzersiz için tek bir .npy dosyası var:
    - shape: (num_images, 132)
    - Her satır bir görselin keypoint'lerini temsil eder
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: data/processed klasörü yolu
            split: 'train' veya 'test'
            transform: Veri dönüşümü (opsiyonel)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Keypoint dosyalarını yükle
        images_dir = self.data_dir / split / "images"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Klasör bulunamadı: {images_dir}")
        
        # Tüm .npy dosyalarını bul
        keypoint_files = sorted(list(images_dir.glob("*_keypoints.npy")))
        
        if not keypoint_files:
            raise ValueError(f"Keypoint dosyası bulunamadı: {images_dir}")
        
        # Verileri ve etiketleri yükle
        self.data = []
        self.labels = []
        self.exercise_names = []
        
        for keypoint_file in keypoint_files:
            # Egzersiz adını dosya adından çıkar
            # Örnek: "squat_keypoints.npy" -> "squat"
            exercise_name = keypoint_file.stem.replace("_keypoints", "")
            # Normalize et
            exercise_name = normalize_exercise_name(exercise_name)
            
            # Keypoint'leri yükle
            keypoints = np.load(keypoint_file)  # shape: (num_images, 132)
            
            # Her görsel için etiket ekle
            num_images = len(keypoints)
            self.data.append(keypoints)
            self.labels.extend([exercise_name] * num_images)
            self.exercise_names.append(exercise_name)
        
        # Tüm verileri birleştir
        self.data = np.vstack(self.data)  # shape: (total_images, 132)
        
        # Label encoder oluştur
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Sınıf sayısı
        self.num_classes = len(self.label_encoder.classes_)
        self.class_names = self.label_encoder.classes_
        
        print(f"\n📊 {split.upper()} Görsel Veri Seti:")
        print(f"   Toplam görsel: {len(self.data)}")
        print(f"   Egzersiz sayısı: {self.num_classes}")
        print(f"   Keypoint boyutu: {self.data.shape[1]}")
        print(f"   Egzersizler: {', '.join(self.class_names)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        keypoints = self.data[idx]
        label = self.labels_encoded[idx]
        
        # NumPy array'den PyTorch tensor'e çevir
        keypoints = torch.FloatTensor(keypoints)
        label = torch.LongTensor([label])[0]  # Scalar tensor
        
        if self.transform:
            keypoints = self.transform(keypoints)
        
        return keypoints, label


class ExerciseVideoDataset(Dataset):
    """
    Videolardan çıkarılmış keypoint'ler için Dataset
    
    Her video için ayrı bir .npy dosyası var:
    - shape: (frame_count, 132)
    - Her satır bir frame'in keypoint'lerini temsil eder
    """
    
    def __init__(self, data_dir, split='train', sequence_length=60, transform=None):
        """
        Args:
            data_dir: data/processed klasörü yolu
            split: 'train' veya 'test'
            sequence_length: Sequence uzunluğu (sliding window için)
            transform: Veri dönüşümü (opsiyonel)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Video keypoint dosyalarını yükle
        videos_dir = self.data_dir / split / "videos"
        
        if not videos_dir.exists():
            raise FileNotFoundError(f"Klasör bulunamadı: {videos_dir}")
        
        # Tüm .npy dosyalarını bul
        keypoint_files = sorted(list(videos_dir.glob("*_keypoints.npy")))
        
        if not keypoint_files:
            raise ValueError(f"Keypoint dosyası bulunamadı: {videos_dir}")
        
        # Verileri ve etiketleri yükle
        self.sequences = []
        self.labels = []
        self.video_names = []
        
        for keypoint_file in keypoint_files:
            # Egzersiz adını dosya adından çıkar
            # Örnek: "squat_1_keypoints.npy" -> "squat"
            video_name = keypoint_file.stem.replace("_keypoints", "")
            
            # Egzersiz adını bul (dosya adından)
            # Format: "<exercise>_<number>_keypoints.npy"
            parts = video_name.split("_")
            exercise_name = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            # Normalize et
            exercise_name = normalize_exercise_name(exercise_name)
            
            # Keypoint'leri yükle
            keypoints = np.load(keypoint_file)  # shape: (frame_count, 132)
            
            # Sequence'lere böl (sliding window)
            if len(keypoints) >= sequence_length:
                num_sequences = len(keypoints) - sequence_length + 1
                for i in range(num_sequences):
                    sequence = keypoints[i:i+sequence_length]
                    self.sequences.append(sequence)
                    self.labels.append(exercise_name)
                    self.video_names.append(video_name)
            else:
                # Video çok kısa, padding yap
                padding = np.zeros((sequence_length - len(keypoints), keypoints.shape[1]))
                sequence = np.vstack([keypoints, padding])
                self.sequences.append(sequence)
                self.labels.append(exercise_name)
                self.video_names.append(video_name)
        
        # Label encoder oluştur
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Sınıf sayısı
        self.num_classes = len(self.label_encoder.classes_)
        self.class_names = self.label_encoder.classes_
        
        print(f"\n📹 {split.upper()} Video Veri Seti:")
        print(f"   Toplam sequence: {len(self.sequences)}")
        print(f"   Egzersiz sayısı: {self.num_classes}")
        print(f"   Sequence uzunluğu: {self.sequence_length}")
        print(f"   Keypoint boyutu: {self.sequences[0].shape[1]}")
        print(f"   Egzersizler: {', '.join(self.class_names)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels_encoded[idx]
        
        # NumPy array'den PyTorch tensor'e çevir
        sequence = torch.FloatTensor(sequence)
        label = torch.LongTensor([label])[0]  # Scalar tensor
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label


def create_data_loaders(
    data_dir='data/processed',
    batch_size=32,
    val_ratio=0.2,
    sequence_length=60,
    num_workers=0
):
    """
    Train ve validation DataLoader'ları oluştur
    
    Args:
        data_dir: data/processed klasörü yolu
        batch_size: Batch boyutu
        val_ratio: Validation oranı
        sequence_length: Video sequence uzunluğu
        num_workers: DataLoader worker sayısı
    
    Returns:
        train_loader_images, val_loader_images, train_loader_videos, val_loader_videos,
        num_classes, class_names
    """
    data_path = Path(data_dir)
    
    # Görsel veri setleri
    print("\n" + "="*60)
    print("GORSEL VERI SETLERI YUKLENIYOR")
    print("="*60)
    
    full_image_dataset = ExerciseImageDataset(data_path, split='train')
    image_class_names = full_image_dataset.class_names
    
    # Train/Validation split
    indices = np.arange(len(full_image_dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_ratio, random_state=42, shuffle=True
    )
    
    # Subset oluştur
    train_image_dataset = torch.utils.data.Subset(full_image_dataset, train_indices)
    val_image_dataset = torch.utils.data.Subset(full_image_dataset, val_indices)
    
    # DataLoader'lar
    train_loader_images = DataLoader(
        train_image_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader_images = DataLoader(
        val_image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\n[OK] Gorsel veri setleri hazir:")
    print(f"   Train: {len(train_image_dataset)} gorsel")
    print(f"   Validation: {len(val_image_dataset)} gorsel")
    
    # Video veri setleri
    print("\n" + "="*60)
    print("VIDEO VERI SETLERI YUKLENIYOR")
    print("="*60)
    
    full_video_dataset = ExerciseVideoDataset(
        data_path, split='train', sequence_length=sequence_length
    )
    video_class_names = full_video_dataset.class_names
    
    # Görsel ve video sınıf isimlerini birleştir ve ortak sınıfları kullan
    # Video veri setindeki sınıfları görsel veri setindeki sınıflarla eşleştir
    common_classes = sorted(set(image_class_names) & set(video_class_names))
    
    if len(common_classes) != len(image_class_names):
        print(f"\n[UYARI] Sinif sayisi uyumsuz!")
        print(f"   Gorsel sinif sayisi: {len(image_class_names)}")
        print(f"   Video sinif sayisi: {len(video_class_names)}")
        print(f"   Ortak sinif sayisi: {len(common_classes)}")
        print(f"   Eksik siniflar (gorsel): {set(image_class_names) - set(video_class_names)}")
        print(f"   Eksik siniflar (video): {set(video_class_names) - set(image_class_names)}")
        print(f"\n[UYARI] Sadece ortak siniflar kullanilacak!")
    
    # Ortak sınıfları kullan
    num_classes = len(common_classes)
    class_names = np.array(common_classes)
    
    # Video veri setindeki etiketleri görsel dataset'teki label encoder ile yeniden kodla
    # Görsel dataset'teki label encoder'ı kullan
    image_label_encoder = full_image_dataset.label_encoder
    
    # Video dataset'teki etiketleri yeniden kodla
    valid_indices = []
    new_labels_encoded = []
    
    for idx in range(len(full_video_dataset.labels)):
        original_label = full_video_dataset.labels[idx]
        if original_label in common_classes:
            # Ortak sınıf, görsel dataset'teki label encoder ile kodla
            try:
                new_label = image_label_encoder.transform([original_label])[0]
                new_labels_encoded.append(new_label)
                valid_indices.append(idx)
            except ValueError:
                # Sınıf bulunamadı, atla
                pass
        else:
            # Ortak sınıf değil, atla
            pass
    
    # Sadece geçerli örnekleri kullan
    if len(valid_indices) < len(full_video_dataset):
        print(f"\n[UYARI] {len(full_video_dataset) - len(valid_indices)} gecersiz ornek filtrelendi")
        # Geçerli örnekleri seç
        full_video_dataset.sequences = [full_video_dataset.sequences[i] for i in valid_indices]
        full_video_dataset.labels = [full_video_dataset.labels[i] for i in valid_indices]
        full_video_dataset.labels_encoded = np.array(new_labels_encoded)
        full_video_dataset.num_classes = len(common_classes)
        full_video_dataset.class_names = np.array(common_classes)
    
    # Train/Validation split
    indices = np.arange(len(full_video_dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_ratio, random_state=42, shuffle=True
    )
    
    # Subset oluştur
    train_video_dataset = torch.utils.data.Subset(full_video_dataset, train_indices)
    val_video_dataset = torch.utils.data.Subset(full_video_dataset, val_indices)
    
    # DataLoader'lar
    train_loader_videos = DataLoader(
        train_video_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader_videos = DataLoader(
        val_video_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\n[OK] Video veri setleri hazir:")
    print(f"   Train: {len(train_video_dataset)} sequence")
    print(f"   Validation: {len(val_video_dataset)} sequence")
    print(f"   Ortak sinif sayisi: {num_classes}")
    
    return (
        train_loader_images, val_loader_images,
        train_loader_videos, val_loader_videos,
        num_classes, class_names
    )


if __name__ == "__main__":
    # Çalışma dizinini proje kök dizinine ayarla
    import os
    os.chdir(project_root)
    
    # Test
    loaders = create_data_loaders(batch_size=16)
    train_img, val_img, train_vid, val_vid, num_classes, class_names = loaders
    
    print(f"\n{'='*60}")
    print("VERI YUKLEME TESTI")
    print(f"{'='*60}")
    print(f"\nSinif sayisi: {num_classes}")
    print(f"Sinif isimleri: {class_names}")
    
    # Bir batch test et
    print("\n[Gorsel batch testi]")
    keypoints, labels = next(iter(train_img))
    print(f"   Keypoints shape: {keypoints.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Ornek label: {labels[0].item()} -> {class_names[labels[0].item()]}")
    
    print("\n[Video batch testi]")
    sequences, labels = next(iter(train_vid))
    print(f"   Sequences shape: {sequences.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Ornek label: {labels[0].item()} -> {class_names[labels[0].item()]}")

