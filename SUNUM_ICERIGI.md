# YAPAY ZEKA DESTEKLİ EGZERSİZ SINIFLANDIRMA SİSTEMİ
## Sunum İçeriği

---

## 1. PROBLEM TANIMI VE AMAÇ BELİRLEME

### Problem
- Egzersiz videolarından otomatik olarak egzersiz türünü tanıma ihtiyacı
- Kişisel antrenör uygulamaları için otomatik form analizi
- Spor salonlarında egzersiz takibi ve geri bildirim

### Proje Amacı
- **22 farklı egzersiz türünü** video ve görsellerden otomatik olarak sınıflandırmak
- MediaPipe pose detection ile vücut pozisyonlarını tespit etmek
- Derin öğrenme modelleri ile egzersiz tanıma yapmak

### Yapılan İşlemler (Kısa Özet)
1. ✅ Kaggle'dan açık veri seti indirildi
2. ✅ Videolar train/test olarak bölündü (%70/%30)
3. ✅ Videolardan frame'ler çıkarıldı
4. ✅ MediaPipe ile keypoints (vücut noktaları) çıkarıldı
5. ✅ İki farklı model eğitildi:
   - **Image Classifier**: Görsellerden egzersiz tanıma (MLP)
   - **Sequence Classifier**: Videolardan egzersiz tanıma (LSTM)
6. ✅ Modeller değerlendirildi ve metrikler hesaplandı

---

## 2. KULLANILAN TEKNOLOJİLER

### Derin Öğrenme Framework
- **PyTorch 2.4.0** (CUDA 12.1 desteği ile GPU hızlandırma)

### Pose Detection
- **MediaPipe 0.10.9** (Google'ın açık kaynak pose detection kütüphanesi)
  - 33 vücut landmark noktası tespiti
  - Her landmark için: x, y, z koordinatları + visibility skoru
  - Toplam: 33 × 4 = **132 özellik**

### Veri İşleme
- **OpenCV 4.8.1.78**: Video/görsel okuma ve işleme
- **NumPy**: Sayısal hesaplamalar (MediaPipe uyumlu <2.0.0)
- **scikit-learn**: Veri bölme, metrik hesaplama, label encoding

### Görselleştirme ve Analiz
- **matplotlib**: Grafik çizimi
- **seaborn**: Confusion matrix görselleştirme
- **pandas**: Veri analizi

### Diğer
- **tqdm**: İlerleme çubuğu
- **json**: Model ve sonuç kayıtları

---

## 3. VERİ TOPLAMA AŞAMALARI

### Veri Seti Kaynağı
- **Kaggle**: Açık kaynak egzersiz video veri seti
- Veri seti içeriği: Farklı egzersiz türlerine ait videolar

### Veri Seti Özellikleri
- **22 egzersiz sınıfı**:
  1. barbell biceps curl
  2. bench press
  3. chest fly machine
  4. deadlift
  5. decline bench press
  6. hammer curl
  7. hip thrust
  8. incline bench press
  9. lat pulldown
  10. lateral raise
  11. leg extension
  12. leg raises
  13. plank
  14. pull Up
  15. push-up
  16. romanian deadlift
  17. russian twist
  18. shoulder press
  19. squat
  20. t bar row
  21. tricep dips
  22. tricep Pushdown

### Veri Yapısı
- **Ham veri**: `data/raw_data/videos/` (her egzersiz için ayrı klasör)
- **İşlenmiş veri**: 
  - Train: `data/train/videos/` ve `data/train/images/`
  - Test: `data/test/videos/`
  - Keypoints: `data/processed/train/` ve `data/processed/test/`

---

## 4. VERİ İŞLEME AŞAMALARI

### 4.1. Train/Test Split
**Script**: `src/util/train_test_split.py`
- Ham videoları %70 train, %30 test olarak bölme
- Random seed (42) ile tekrarlanabilir split
- Her egzersiz için ayrı klasör yapısı oluşturma

**Çıktı**:
- `data/train/videos/{egzersiz}/` → Train videoları
- `data/test/videos/{egzersiz}/` → Test videoları

### 4.2. Frame Çıkarma
**Script**: `src/util/extract_frames.py`
- Train videolarından görsel çıkarma
- **Parametreler**:
  - Hedef FPS: 6 (video başına ~6 frame/saniye)
  - Maksimum frame: 400 frame/video
- Frame'ler görsel dosyaları olarak kaydedilir

**Çıktı**: `data/train/images/{egzersiz}/` → Her video için frame'ler

### 4.3. Keypoints Çıkarma
**Script**: `src/data_collector_keypoints.py`
- MediaPipe ile her frame/video'dan vücut pozisyonu tespiti
- **Her landmark için**:
  - x, y, z koordinatları (0-1 arası normalize)
  - visibility skoru (0-1 arası)
- **Toplam özellik**: 33 landmark × 4 = 132 boyut

**Çıktı**:
- Görseller: `data/processed/train/images/{egzersiz}_keypoints.npy` (shape: num_images, 132)
- Videolar: `data/processed/train/videos/{video}_keypoints.npy` (shape: frame_count, 132)
- Test: `data/processed/test/videos/{video}_keypoints.npy`

### 4.4. Veri Ön İşleme (Preprocessing)

#### ✅ Yapılan İşlemler:

1. **Keypoints Normalizasyonu**
   - MediaPipe zaten keypoints'leri normalize ediyor
   - x, y, z koordinatları: 0-1 arası (görüntü boyutuna göre)
   - visibility: 0-1 arası (görünürlük skoru)

2. **Label Encoding**
   - Egzersiz isimlerini sayısal etiketlere çevirme
   - scikit-learn LabelEncoder kullanıldı
   - 22 sınıf → 0-21 arası sayısal etiketler

3. **Train/Validation Split**
   - Train verisinden %20 validation seti ayrıldı
   - Random seed (42) ile tekrarlanabilir

4. **Sequence Padding** (Video için)
   - Kısa videolar için zero-padding
   - Sequence length: 60 frame (sliding window)

5. **Sliding Window** (Video için)
   - Uzun videoları 60 frame'lik sequence'lere bölme
   - Overlap ile daha fazla örnek oluşturma

#### ⚠️ Yapılmayan İşlemler:

1. **Veri Arttırma (Data Augmentation)**
   - Rotation, scale, noise ekleme gibi işlemler yapılmadı
   - Transform parametresi hazır ama kullanılmadı
   - **Neden?**: MediaPipe keypoints zaten normalize ve robust
   - **Gelecek geliştirme**: Augmentation eklenebilir

2. **Ek Özellik Çıkarımı**
   - Açı hesaplama (joint angles) yapılmadı
   - Mesafe özellikleri eklenmedi
   - Hız hesaplama (velocity) yapılmadı
   - **Not**: Sadece MediaPipe keypoints kullanıldı (132 boyut)

---

## 5. MODEL MİMARİSİ

### 5.1. Image Classifier (Görsel Modeli)

**Mimari Tip**: MLP (Multi-Layer Perceptron)

**Girdi**: `(batch_size, 132)` - Tek frame keypoints

**Mimari Detayları**:
```
Input Layer: 132 (keypoints)
    ↓
Hidden Layer 1: 256 + ReLU + Dropout(0.3)
    ↓
Hidden Layer 2: 128 + ReLU + Dropout(0.3)
    ↓
Hidden Layer 3: 64 + ReLU + Dropout(0.3)
    ↓
Output Layer: 22 (egzersiz sınıfları)
```

**Kullanım Amacı**: 
- Statik pozlardan (görsellerden) egzersiz tanıma
- Her görsel bağımsız bir örnek olarak kullanılır

**Parametre Sayısı**: ~100K parametre

---

### 5.2. Sequence Classifier (Video Modeli)

**Mimari Tip**: LSTM (Long Short-Term Memory)

**Girdi**: `(batch_size, sequence_length, 132)` - Video sequence

**Mimari Detayları**:
```
Input: (batch, 60, 132) - 60 frame'lik sequence
    ↓
LSTM Layer 1: 128 hidden units, 2 layers
    ↓
Dropout(0.3)
    ↓
LSTM Layer 2: 128 hidden units
    ↓
Dense Layer: 64 + ReLU
    ↓
Dropout(0.3)
    ↓
Output Layer: 22 (egzersiz sınıfları)
```

**Kullanım Amacı**:
- Video sequence'lerinden egzersiz tanıma
- Zaman serisi bilgisini kullanır
- Hareket akışını öğrenir

**Parametre Sayısı**: ~200K parametre

---

## 6. EĞİTİM SÜRECİ

### 6.1. Eğitim Parametreleri

**Optimizer**: Adam
- Learning Rate: 0.001
- Beta1: 0.9, Beta2: 0.999

**Loss Fonksiyonu**: CrossEntropyLoss
- Çok sınıflı sınıflandırma için uygun

**Learning Rate Scheduler**: ReduceLROnPlateau
- Patience: 5 epoch
- Factor: 0.5 (yarıya indir)
- Validation loss'a göre otomatik ayarlama

**Epoch Sayısı**: 50
- Early stopping yok (tüm epoch'lar tamamlandı)

**Batch Size**:
- Image Model: 32
- Sequence Model: 16 (daha büyük bellek kullanımı)

**Validation Ratio**: 0.2 (%20 validation, %80 train)

### 6.2. Eğitim Süreci

**Adımlar**:
1. Veri yükleme (DataLoader)
2. Her epoch için:
   - Train: Forward pass → Loss → Backward pass → Update weights
   - Validation: Forward pass → Loss (weights güncellenmez)
   - En iyi validation accuracy modeli kaydedilir
3. Learning rate scheduler güncellenir
4. Eğitim geçmişi kaydedilir

**Çıktılar**:
- `models/{model_type}/best_model.pth` - En iyi validation accuracy
- `models/{model_type}/final_model.pth` - Son epoch modeli
- `models/{model_type}/training_history.json` - Loss ve accuracy geçmişi
- `models/{model_type}/class_names.json` - Sınıf isimleri

### 6.3. Eğitim Sonuçları ve Grafikler

**Grafik Oluşturma**:
- Eğitim grafikleri `training_history.json` dosyalarından oluşturulur
- Script: `src/util/visualize_training.py`
- Komut: `python src/util/visualize_training.py --model_type both`

**Grafik İçeriği**:
- **Sol Grafik: Loss (Kayıp)**
  - Train Loss ve Validation Loss
  - Epoch ilerledikçe aşağı doğru iner (loss azalır)
  - Overfitting kontrolü için train/val loss farkına bakılır
  
- **Sağ Grafik: Accuracy (Doğruluk)**
  - Train Accuracy ve Validation Accuracy
  - Epoch ilerledikçe yukarı doğru çıkar (accuracy artar)
  - Model performansını gösterir

**Çıktı Dosyaları**:
- `outputs/training_curves/image_classifier_training_curves.png`
- `outputs/training_curves/sequence_classifier_training_curves.png`

**Image Model Sonuçları**:
- Train/Validation loss azalışı gözlemlendi
- Validation accuracy: ~91% (son epoch)
- Overfitting kontrolü: Dropout ile önlendi
- 50 epoch boyunca düzenli iyileşme

**Sequence Model Sonuçları**:
- LSTM ile zaman serisi öğrenimi
- Sequence uzunluğu: 60 frame
- Validation accuracy: ~99% (son epoch)
- Video sequence'lerinden öğrenme
- Çok hızlı yakınsama (ilk 10 epoch'ta %95+)

---

## 7. DEĞERLENDİRME

### 7.1. Değerlendirme Metrikleri

**Temel Metrikler**:
- **Accuracy**: Doğru tahmin oranı
- **Precision**: Pozitif tahminlerin doğruluk oranı
- **Recall**: Gerçek pozitiflerin tespit oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması

**Görselleştirmeler**:
- **Confusion Matrix**: Hangi sınıfların karıştırıldığını gösterir
- **Classification Report**: Sınıf bazlı detaylı metrikler

### 7.2. Değerlendirme Sonuçları

**Image Model Sonuçları**:
- 22 egzersiz sınıfı için değerlendirme yapıldı
- Sınıf bazlı F1-Score örnekleri:
  - "decline bench press": 0.95
  - "chest fly machine": 0.94
  - "bench press": 0.88
  - "barbell biceps curl": 0.82
  - "deadlift": 0.82

**Sequence Model Sonuçları**:
- Test videoları üzerinde değerlendirme
- Video sequence'lerinden tahmin
- Zaman serisi bilgisi kullanıldı

**Çıktı Dosyaları**:
- `outputs/evaluation/image_model_confusion_matrix.png`
- `outputs/evaluation/image_model_classification_report.json`
- `outputs/evaluation/sequence_model_confusion_matrix.png`
- `outputs/evaluation/sequence_model_classification_report.json`

---

## 8. ELDE EDİLEN ÇIKTILAR

### 8.1. Eğitilmiş Modeller

✅ **Image Classifier (MLP)**
- Görsellerden egzersiz tanıma
- 22 sınıf için eğitilmiş
- Best model ve final model kaydedildi

✅ **Sequence Classifier (LSTM)**
- Videolardan egzersiz tanıma
- 22 sınıf için eğitilmiş
- Best model ve final model kaydedildi

### 8.2. Değerlendirme Sonuçları

✅ **Confusion Matrix Görselleri**
- Hangi egzersizlerin karıştırıldığını gösterir
- Model performansını görselleştirir

✅ **Sınıf Bazlı Detaylı Raporlar (JSON)**
- Her egzersiz için precision, recall, F1-score
- Support (örnek sayısı) bilgisi

### 8.3. İşlenmiş Veri Seti

✅ **Keypoints Dosyaları (.npy)**
- Train görseller: `data/processed/train/images/`
- Train videolar: `data/processed/train/videos/`
- Test videolar: `data/processed/test/videos/`
- Toplam: 22 egzersiz için hazır veri seti

### 8.4. Eğitim Geçmişi

✅ **Training History (JSON)**
- Her epoch için train/validation loss
- Her epoch için train/validation accuracy
- Model gelişimini takip etmek için

---

## ÖZET VE SONUÇLAR

### Başarılar
✅ 22 egzersiz sınıfı için sınıflandırma modelleri eğitildi
✅ MediaPipe ile robust pose detection yapıldı
✅ İki farklı yaklaşım (görsel + video) uygulandı
✅ Detaylı değerlendirme metrikleri hesaplandı

### Gelecek Geliştirmeler
🔮 Veri arttırma (data augmentation) eklenebilir
🔮 Ek özellik çıkarımı (açı, mesafe, hız) yapılabilir
🔮 Form analizi ve geri bildirim eklenebilir
🔮 Hibrit model (görsel + video birleştirme) geliştirilebilir

---

## TEKNİK DETAYLAR

### Veri Ön İşleme Özeti
- ✅ Keypoints normalizasyonu (MediaPipe otomatik)
- ✅ Label encoding
- ✅ Train/validation split
- ✅ Sequence padding ve sliding window
- ⚠️ Veri arttırma yapılmadı (gelecek geliştirme)

### Model Özeti
- **Image Model**: MLP (132 → 256 → 128 → 64 → 22)
- **Sequence Model**: LSTM (60×132 → 128 → 64 → 22)

### Performans
- Her iki model de başarıyla eğitildi
- Sınıf bazlı detaylı metrikler hesaplandı
- Confusion matrix ile görselleştirme yapıldı

