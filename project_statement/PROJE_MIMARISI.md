# Proje Mimarisi ve Veri Seti Şeması

## 📋 Genel Yaklaşım

- **Eğitim (Train)**: Görseller + Videolar (hibrit yaklaşım)
- **Test**: Sadece Videolar (gerçekçi senaryo)
- **Hedef**: Egzersiz form analizi ve kas grubu görselleştirmesi

---

## 📁 Veri Seti Yapısı

### Ana Klasör Yapısı

```
data/
├── train/                        # Eğitim verisi
│   ├── images/                   # Görseller (statik pozlar)
│   │   ├── squat/
│   │   │   ├── correct/
│   │   │   │   ├── squat_correct_001.jpg
│   │   │   │   ├── squat_correct_002.jpg
│   │   │   │   └── ... (100+ görsel)
│   │   │   └── incorrect/
│   │   │       ├── knee_cave/
│   │   │       │   ├── squat_incorrect_knee_cave_001.jpg
│   │   │       │   └── ...
│   │   │       ├── forward_lean/
│   │   │       │   └── squat_incorrect_lean_001.jpg
│   │   │       └── shallow/
│   │   │           └── squat_incorrect_shallow_001.jpg
│   │   ├── pushup/
│   │   │   ├── correct/
│   │   │   └── incorrect/
│   │   │       ├── high_hips/
│   │   │       ├── low_hips/
│   │   │       └── shallow/
│   │   ├── lunge/
│   │   │   └── ...
│   │   └── ... (diğer egzersizler)
│   │
│   └── videos/                   # Eğitim videoları (tam hareketler)
│       ├── squat/
│       │   ├── squat_train_001.mp4
│       │   ├── squat_train_002.mp4
│       │   └── ... (50+ video)
│       ├── pushup/
│       │   ├── pushup_train_001.mp4
│       │   └── ... (50+ video)
│       ├── lunge/
│       │   └── ...
│       └── ... (diğer egzersizler)
│
├── test/                         # Test verisi
│   └── videos/                   # SADECE test videoları
│       ├── squat/
│       │   ├── squat_test_001.mp4
│       │   ├── squat_test_002.mp4
│       │   └── ... (20+ video)
│       ├── pushup/
│       │   ├── pushup_test_001.mp4
│       │   └── ... (20+ video)
│       ├── lunge/
│       │   └── ...
│       └── ... (diğer egzersizler)
│
├── processed/                    # İşlenmiş keypoints ve özellikler
│   ├── train/
│   │   ├── images/               # Görsel keypoints
│   │   │   ├── squat_correct.npy        # (num_images, 132)
│   │   │   ├── squat_incorrect.npy      # (num_images, 132)
│   │   │   ├── pushup_correct.npy
│   │   │   └── ... (her egzersiz için)
│   │   │
│   │   └── videos/               # Video keypoints (sequence)
│   │       ├── squat_train_001.npy      # (frame_count, 132)
│   │       ├── squat_train_002.npy
│   │       └── ... (her video için)
│   │
│   └── test/
│       └── videos/               # Test video keypoints
│           ├── squat_test_001.npy       # (frame_count, 132)
│           ├── squat_test_002.npy
│           └── ... (her test video için)
│
└── labels/                       # Etiketler ve metadata
    ├── train_images_labels.json  # Görsel etiketleri
    │   {
    │     "squat_correct_001.jpg": {
    │       "exercise": "squat",
    │       "form": "correct",
    │       "error_type": null,
    │       "source_video": "squat_train_001.mp4"
    │     },
    │     "squat_incorrect_knee_cave_001.jpg": {
    │       "exercise": "squat",
    │       "form": "incorrect",
    │       "error_type": "knee_cave",
    │       "source_video": "squat_train_002.mp4"
    │     }
    │   }
    │
    ├── train_videos_labels.json  # Train video etiketleri
    │   {
    │     "squat_train_001.mp4": {
    │       "exercise": "squat",
    │       "form_scores": [0.95, 0.92, 0.98, ...],  # Frame-by-frame
    │       "overall_score": 0.95,
    │       "error_types": []
    │     }
    │   }
    │
    ├── test_videos_labels.json   # Test video etiketleri
    │   {
    │     "squat_test_001.mp4": {
    │       "exercise": "squat",
    │       "form_scores": [0.88, 0.85, 0.90, ...],
    │       "overall_score": 0.88
    │     }
    │   }
    │
    └── exercise_metadata.json    # Genel metadata
        {
          "exercises": ["squat", "pushup", "lunge", "plank", ...],
          "error_types": {
            "squat": ["knee_cave", "forward_lean", "shallow"],
            "pushup": ["high_hips", "low_hips", "shallow"],
            "lunge": ["knee_over_toes", "forward_lean"]
          },
          "muscle_groups": {
            "squat": ["quadriceps", "glutes", "hamstrings", "calves", "core"],
            "pushup": ["chest", "triceps", "shoulders", "core"]
          }
        }
```

---

## 🏗️ Proje Modül Mimarisi

### Kaynak Kod Yapısı

```
src/
├── pose_detector.py              # ✅ Mevcut: Temel pose detection
│   ├── PoseDetector class
│   ├── process_frame()           # Tek frame işleme
│   ├── process_image()           # 🔨 Görsel işleme (eklenecek)
│   ├── process_video()           # ✅ Video işleme (mevcut)
│   ├── extract_keypoints()       # ✅ Keypoints çıkarımı
│   └── draw_pose()               # ✅ Görselleştirme
│
├── data_collector.py             # 🔨 Yeni: Veri toplama ve işleme
│   ├── DataCollector class
│   ├── process_images_folder()   # Klasördeki tüm görselleri işle
│   ├── process_videos_folder()   # Klasördeki tüm videoları işle
│   ├── create_labels_from_structure()  # Klasör yapısından otomatik etiket
│   └── validate_train_test_split()     # Veri leakage kontrolü
│
├── feature_extractor.py          # 🔨 Yeni: Gelişmiş özellik çıkarımı
│   ├── FeatureExtractor class
│   ├── extract_angles()          # Açı hesaplama (diz, kalça, vb.)
│   ├── extract_distances()       # Mesafe özellikleri
│   ├── extract_velocity()        # Hız hesaplama (video için)
│   ├── normalize_keypoints()     # Normalizasyon (kişi boyutuna göre)
│   └── combine_features()        # Tüm özellikleri birleştir
│
├── data_preprocessor.py          # 🔨 Yeni: Veri hazırlama ve augmentation
│   ├── DataPreprocessor class
│   ├── create_sequences()        # Görsellerden sequence oluştur (opsiyonel)
│   ├── split_train_val()         # Train/validation split
│   ├── augment_data()            # Data augmentation (rotation, scale, noise)
│   └── prepare_for_training()    # Model için hazır hale getir
│
├── models/
│   ├── __init__.py
│   │
│   ├── exercise_classifier.py    # 🔨 Yeni: Egzersiz tanıma modeli
│   │   ├── ImageClassifier       # Tek frame modeli (CNN/MLP)
│   │   │   └── Input: (batch, 132) → Output: exercise_class
│   │   │
│   │   ├── SequenceClassifier    # Sequence modeli (LSTM)
│   │   │   └── Input: (batch, seq_len, 132) → Output: exercise_class
│   │   │
│   │   └── HybridClassifier      # Hibrit model (Image + Sequence)
│   │       ├── Image branch: görsellerden öğrenme
│   │       ├── Sequence branch: videolardan öğrenme
│   │       └── Fusion layer: birleştirme
│   │
│   ├── form_analyzer_model.py    # 🔨 Yeni: Form analizi modeli
│   │   ├── FormScoreModel        # Form skoru tahmini (0-1)
│   │   ├── ErrorDetectionModel   # Hata türü tespiti
│   │   └── FormAnalyzer          # Kombine analiz
│   │
│   └── trainer.py                # 🔨 Yeni: Model eğitimi
│       ├── Trainer class
│       ├── train_image_model()   # Görsellerle eğitim
│       ├── train_sequence_model() # Videolarla eğitim
│       ├── train_hybrid_model()  # Hibrit eğitim
│       └── evaluate_model()      # Model değerlendirme
│
├── form_analyzer.py              # 🔨 Yeni: Form analizi modülü
│   ├── FormAnalyzer class
│   ├── analyze_frame()           # Tek frame analizi
│   ├── analyze_sequence()        # Video sequence analizi
│   ├── calculate_form_score()    # Form skoru (0-1)
│   ├── detect_errors()           # Hata tespiti
│   └── provide_feedback()        # Geri bildirim oluştur
│
├── muscle_mapper.py              # ✅ Mevcut: Kas grubu mapping
│   ├── get_muscle_groups()       # Egzersiz → kas grupları
│   ├── get_activation_levels()   # Aktivasyon seviyeleri
│   └── get_landmark_indices()    # Landmark eşleme
│
└── visualizer.py                 # 🔨 Yeni: Görselleştirme
    ├── Visualizer class
    ├── draw_muscle_heatmap()     # Kas grubu heatmap
    ├── draw_form_feedback()      # Form geri bildirim görselleştirme
    ├── create_summary_video()    # Özet video oluşturma
    └── generate_report()         # Rapor oluşturma
```

---

## 🔄 Veri İşleme Akışı

### Aşama 1: Veri Toplama ve İşleme

```
1. Görselleri organize et
   data/train/images/squat/correct/*.jpg
   data/train/images/squat/incorrect/knee_cave/*.jpg
   ↓
2. Görsellerden keypoints çıkar
   src/data_collector.py → process_images_folder()
   Her görsel → (132,) keypoints array
   ↓
3. Görsel keypoints kaydet
   data/processed/train/images/squat_correct.npy
   Şekil: (num_images, 132)
   ↓
4. Videoları organize et
   data/train/videos/squat/*.mp4
   data/test/videos/squat/*.mp4
   ↓
5. Videolardan keypoints çıkar
   src/data_collector.py → process_videos_folder()
   Her video → (frame_count, 132) sequence
   ↓
6. Video keypoints kaydet
   data/processed/train/videos/squat_train_001.npy
   data/processed/test/videos/squat_test_001.npy
   ↓
7. Otomatik etiket oluştur
   Klasör yapısından → labels/*.json
```

### Aşama 2: Özellik Çıkarımı ve Hazırlama

```
Görsel Keypoints (num_images, 132)
   ↓
Özellik Çıkarımı
   ├── Açı hesaplama (joint angles)
   ├── Mesafe özellikleri
   └── Normalizasyon
   ↓
Özellik Sayısı: 132 → ~220
   ↓
Data Augmentation (rotation, scale, noise)
   ↓
Train/Validation Split
   ↓
Model Eğitimi Hazır
```

```
Video Keypoints Sequence (frame_count, 132)
   ↓
Özellik Çıkarımı (her frame için)
   ├── Açı hesaplama
   ├── Mesafe özellikleri
   ├── Hız hesaplama (frame-to-frame)
   └── Normalizasyon
   ↓
Sequence Length: 30-60 frame (sliding window)
   ↓
Sequence Format: (num_sequences, seq_len, features)
   ↓
Train/Validation Split
   ↓
Model Eğitimi Hazır
```

---

## 🤖 Model Eğitimi Stratejisi

### Strateji: Hybrid Training

#### Aşama 1: Görsellerle Eğitim (Statik Poz Öğrenme)

```
Input: Görsel keypoints (num_images, 132)
Model: ImageClassifier (CNN veya MLP)
Output: Exercise classification + Form score

Özellikler:
- Hızlı eğitim
- Her görsel bağımsız örnek
- Çok sayıda örnek (100+ / egzersiz)
- Statik pozları öğrenir
```

#### Aşama 2: Videolarla Eğitim (Zaman Serisi Öğrenme)

```
Input: Video sequences (num_sequences, seq_len, 132)
Model: SequenceClassifier (LSTM/GRU)
Output: Exercise classification + Form score

Özellikler:
- Zaman serisi analizi
- Hareket akışı öğrenme
- Görsellerden öğrendiklerini genişletir
```

#### Aşama 3: Hibrit Model (Opsiyonel)

```
Image Branch: Görsellerden öğrenilen özellikler
Sequence Branch: Videolardan öğrenilen özellikler
Fusion: İkisini birleştiren katman

Avantaj: Her iki veri tipinden de öğrenir
```

### Model Mimarisi Örnekleri

#### Image Classifier (Görseller için)

```python
Input: (batch_size, 132)  # Keypoints
   ↓
Dense(256) + ReLU
   ↓
Dropout(0.3)
   ↓
Dense(128) + ReLU
   ↓
Dense(64) + ReLU
   ↓
Output 1: Exercise Classification (softmax)  # squat, pushup, ...
Output 2: Form Score (sigmoid)  # 0-1
```

#### Sequence Classifier (Videolar için)

```python
Input: (batch_size, sequence_length, 132)
   ↓
LSTM(128, return_sequences=True)
   ↓
Dropout(0.3)
   ↓
LSTM(64, return_sequences=False)
   ↓
Dense(32) + ReLU
   ↓
Output 1: Exercise Classification (softmax)
Output 2: Form Score (sigmoid)
```

---

## 📊 Veri Formatı Detayları

### Görsel Keypoints (Eğitim)

```python
# Her görsel için
shape: (132,)  # 33 landmark × 4 değer (x, y, z, visibility)

# Tüm görseller için
shape: (num_images, 132)
Örnek: (500, 132)  # 500 squat görseli

# Her görsel bağımsız bir örnek
# Her frame statik poz olarak kullanılır
```

### Video Keypoints (Eğitim ve Test)

```python
# Her video için sequence
shape: (frame_count, 132)
Örnek: (981, 132)  # 981 frame'lik video

# Zaman serisi olarak kullanılır
# Sequence length: 30-60 frame (sliding window)
# Final shape: (num_sequences, seq_len, 132)
```

### Özellik Vektörü (Feature Extraction Sonrası)

```python
# Temel keypoints: 132
# Açı özellikleri: ~20
# Mesafe özellikleri: ~15
# Hız özellikleri: ~66 (video için)
# Toplam: ~220-250 özellik
```

---

## 🎯 Test ve Değerlendirme

### Test Senaryosu

```
Input: Test videos (data/test/videos/*.mp4)
   ↓
1. Video'dan keypoints çıkar
   (frame_count, 132)
   ↓
2. Sequence'lere böl (sliding window)
   (num_sequences, seq_len, 132)
   ↓
3. Model tahmini
   - Exercise classification
   - Form score (her sequence için)
   ↓
4. Video-level toplama
   - Overall exercise classification
   - Average form score
   - Frame-by-frame analiz
   ↓
5. Değerlendirme
   - Accuracy
   - Precision/Recall
   - Form score korelasyonu
```

### Metrikler

```
Egzersiz Tanıma:
  - Accuracy: Doğru egzersiz tanıma oranı
  - Confusion Matrix: Hangi egzersizler karıştırılıyor

Form Analizi:
  - MSE: Form skoru tahmin hatası
  - Correlation: Gerçek vs. tahmin edilen skor
  - Error Detection Rate: Hata tespit başarı oranı
```

---

## 📝 Önemli Notlar ve Kurallar

### Train/Test Split Kuralları

1. **Veri Kaynağı Kontrolü**
   - Eğer `squat_train_001.mp4` → train'de ise
   - Bu videodan kesilmiş tüm görseller → train/images/ içinde olmalı
   - Test videolarından kesilmiş görseller kullanılmamalı

2. **Veri Leakage Önleme**
   - Aynı videodan görsel + video ikisi de train'de olmalı
   - Test'te sadece video kullanılır, görsel kullanılmaz

3. **Split Oranı**
   - Videolar: %70 train, %30 test
   - Görseller: Train videolarından kesilmiş olanlar → train/images/

### Dosya İsimlendirme Standartları

```
Görseller:
  {exercise}_{form}_{error_type}_{id}.jpg
  Örnek: squat_correct_001.jpg
         squat_incorrect_knee_cave_001.jpg

Videolar (Train):
  {exercise}_train_{id}.mp4
  Örnek: squat_train_001.mp4

Videolar (Test):
  {exercise}_test_{id}.mp4
  Örnek: squat_test_001.mp4
```

---

## 🚀 Geliştirme Aşamaları

### Aşama 1: Veri Hazırlama ✅
- [x] Klasör yapısı oluştur
- [ ] Görselleri organize et
- [ ] Videoları organize et
- [ ] Veri toplama scripti geliştir

### Aşama 2: Keypoints Çıkarımı 🔨
- [ ] Görsellerden keypoints çıkar
- [ ] Videolardan keypoints çıkar
- [ ] Otomatik etiket oluştur
- [ ] Veri doğrulama

### Aşama 3: Özellik Çıkarımı 🔨
- [ ] Açı hesaplama modülü
- [ ] Mesafe özellikleri
- [ ] Hız hesaplama (video)
- [ ] Normalizasyon

### Aşama 4: Model Geliştirme 🔨
- [ ] Image Classifier
- [ ] Sequence Classifier
- [ ] Model eğitimi
- [ ] Model değerlendirme

### Aşama 5: Form Analizi 🔨
- [ ] Form skoru hesaplama
- [ ] Hata tespiti
- [ ] Geri bildirim sistemi

### Aşama 6: Görselleştirme 🔨
- [ ] Kas grubu heatmap
- [ ] Form geri bildirim görselleştirme
- [ ] Video overlay

### Aşama 7: Entegrasyon 🔨
- [ ] Tüm modülleri birleştir
- [ ] Ana uygulama
- [ ] Test ve optimizasyon

---

## 💡 Önemli Notlar

### Görsellerden Eğitim Yaklaşımı

**Avantajlar:**
- ✅ Çok sayıda örnek toplamak kolay
- ✅ Her görsel bağımsız örnek (augmentation kolay)
- ✅ Hızlı eğitim

**Dikkat Edilmesi Gerekenler:**
- ⚠️ Model statik pozları öğrenir
- ⚠️ Video test için sequence handling gerekir
- ⚠️ Görseller farklı açılardan olmalı (çeşitlilik)

### Video Test Yaklaşımı

**Avantajlar:**
- ✅ Gerçekçi test senaryosu
- ✅ Zaman serisi analizi
- ✅ Hareket akışı gözlemlenir
- ✅ Gerçek kullanım koşulları

**Dikkat Edilmesi Gerekenler:**
- ⚠️ Test videoları çeşitli olmalı
- ⚠️ Farklı kişiler, açılar, form kalitesi

---

## 📋 Veri Gereksinimleri

### Minimum Veri Miktarları

```
Her Egzersiz İçin:

Görseller (Train):
  - Correct: 100+ görsel
  - Incorrect: 50+ görsel (3-5 hata türü)

Videolar:
  - Train: 50+ video
  - Test: 20+ video

Toplam:
  - 10 egzersiz için:
    - Görseller: ~1500 görsel
    - Videolar: ~700 video (500 train + 200 test)
```

---

*Bu mimariye göre projeyi adım adım inşa edebiliriz.*

