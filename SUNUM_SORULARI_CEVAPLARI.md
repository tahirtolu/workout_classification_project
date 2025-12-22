# SUNUM İÇİN HAZIRLIK: SORULAR VE CEVAPLAR

## 📋 PROJE GENELİ HAKKINDA SORULAR

### S1: Projenizin amacı nedir?
**Cevap:**
- 22 farklı egzersiz türünü video ve görsellerden otomatik olarak sınıflandırmak
- MediaPipe pose detection ile vücut pozisyonlarını tespit etmek
- Derin öğrenme modelleri ile egzersiz tanıma yapmak
- Kişisel antrenör uygulamaları için otomatik form analizi sağlamak

### S2: Neden bu projeyi seçtiniz?
**Cevap:**
- Spor ve sağlık alanında pratik bir uygulama
- Computer vision ve deep learning teknolojilerini birleştirme fırsatı
- Gerçek dünya problemine çözüm (egzersiz form kontrolü)
- Açık kaynak veri seti kullanılabilirliği

### S3: Projenin en büyük zorlukları nelerdi?
**Cevap:**
- 22 farklı egzersiz sınıfını ayırt etmek (bazıları birbirine benzer)
- Veri seti hazırlama ve keypoints çıkarma sürecinin zaman alması
- İki farklı model tipini (görsel + video) koordine etmek
- MediaPipe ile Windows uyumluluğu sorunları
- Train/test split'te veri leakage önleme

---

## 🔧 TEKNOLOJİLER HAKKINDA SORULAR

### S4: Neden MediaPipe kullandınız?
**Cevap:**
- Google'ın açık kaynak, ücretsiz pose detection kütüphanesi
- 33 vücut landmark noktası tespiti (yüksek doğruluk)
- CPU'da çalışabilir (GPU gerektirmez)
- Kolay entegrasyon ve kullanım
- Gerçek zamanlı işleme desteği

### S5: Neden PyTorch seçtiniz, TensorFlow değil?
**Cevap:**
- Daha esnek ve Pythonic API
- Dynamic computation graph (LSTM için avantajlı)
- CUDA desteği ile GPU hızlandırma
- Aktif topluluk ve dokümantasyon
- Eğitim sırasında daha kolay debug

### S6: İki farklı model (Image ve Sequence) neden kullandınız?
**Cevap:**
- **Image Classifier (MLP)**: Statik pozları öğrenir, hızlı eğitim
- **Sequence Classifier (LSTM)**: Zaman serisi bilgisini kullanır, hareket akışını öğrenir
- Hibrit yaklaşım: Her iki veri tipinden de öğrenme
- Gerçek dünyada video kullanıldığı için Sequence Model daha uygun
- Image Model eğitim verisi artırmak için kullanıldı

### S7: Neden LSTM kullandınız, Transformer değil?
**Cevap:**
- LSTM daha az parametre gerektirir (daha hızlı eğitim)
- Sequence length (60 frame) için yeterli
- Transformer daha fazla veri gerektirir
- LSTM video sequence'leri için klasik ve etkili çözüm
- Proje kapsamına uygun (basit ama etkili)

### S8: OpenCV'nin rolü nedir?
**Cevap:**
- Video okuma ve yazma (cv2.VideoCapture)
- Görsel okuma (cv2.imread)
- Frame işleme
- Video özelliklerini alma (FPS, boyut, frame sayısı)
- MediaPipe ile entegrasyon için gerekli

---

## 📊 VERİ İŞLEME HAKKINDA SORULAR

### S9: Veri setinizi nereden aldınız?
**Cevap:**
- Kaggle'dan açık kaynak egzersiz video veri seti
- 22 farklı egzersiz türü
- Her egzersiz için çok sayıda video
- Açık lisanslı, akademik kullanım için uygun

### S10: Train/test split oranınız nedir ve neden?
**Cevap:**
- %70 train, %30 test
- Standart makine öğrenmesi pratiği
- Yeterli eğitim verisi sağlar
- Test seti için yeterli örnek
- Random seed (42) ile tekrarlanabilir

### S11: Neden train'de hem görseller hem videolar var, test'te sadece videolar?
**Cevap:**
- **Train'de**: İki model için veri (Image + Sequence)
- **Test'te**: Gerçek dünya senaryosu (kullanıcı video yükler)
- Veri leakage önleme (test videolarından görsel çıkarılmadı)
- Train'de daha fazla veri = daha iyi öğrenme
- Test gerçekçi kalır

### S12: Keypoints çıkarma işlemi nasıl çalışıyor?
**Cevap:**
- MediaPipe her frame'de 33 vücut landmark noktası tespit eder
- Her landmark için: x, y, z koordinatları + visibility skoru
- Toplam: 33 × 4 = 132 boyut
- Koordinatlar 0-1 arası normalize (görüntü boyutuna göre)
- `pose_detector.py` dosyasında `extract_keypoints()` fonksiyonu

### S13: Veri ön işleme yaptınız mı?
**Cevap:**
- ✅ Keypoints normalizasyonu (MediaPipe otomatik yapıyor)
- ✅ Label encoding (egzersiz isimleri → sayısal etiketler)
- ✅ Train/validation split (%80/%20)
- ✅ Sequence padding (kısa videolar için)
- ✅ Sliding window (uzun videoları 60 frame'lik sequence'lere bölme)
- ❌ Veri arttırma yapılmadı (gelecek geliştirme)

### S14: Neden veri arttırma yapmadınız?
**Cevap:**
- MediaPipe keypoints zaten normalize ve robust
- Keypoints üzerinde rotation/scale augmentation zor
- Yeterli veri seti mevcut
- Gelecek geliştirme olarak eklenebilir
- Öncelik model mimarisi ve eğitim sürecine verildi

---

## 🏗️ MODEL MİMARİSİ HAKKINDA SORULAR

### S15: Image Classifier mimarisini açıklar mısınız?
**Cevap:**
- **Tip**: MLP (Multi-Layer Perceptron)
- **Girdi**: (batch_size, 132) - Tek frame keypoints
- **Katmanlar**:
  - Input: 132
  - Hidden 1: 256 + ReLU + Dropout(0.3)
  - Hidden 2: 128 + ReLU + Dropout(0.3)
  - Hidden 3: 64 + ReLU + Dropout(0.3)
  - Output: 22 (egzersiz sınıfları)
- **Parametre sayısı**: ~100K
- **Kullanım**: Statik pozlardan egzersiz tanıma

### S16: Sequence Classifier mimarisini açıklar mısınız?
**Cevap:**
- **Tip**: LSTM (Long Short-Term Memory)
- **Girdi**: (batch_size, 60, 132) - 60 frame'lik sequence
- **Katmanlar**:
  - LSTM Layer 1: 128 hidden units, 2 layers
  - Dropout(0.3)
  - Dense: 64 + ReLU
  - Dropout(0.3)
  - Output: 22 (egzersiz sınıfları)
- **Parametre sayısı**: ~200K
- **Kullanım**: Video sequence'lerinden egzersiz tanıma

### S17: Neden Dropout kullandınız?
**Cevap:**
- Overfitting önleme
- Model genelleştirme yeteneğini artırır
- %30 dropout oranı (0.3) standart değer
- Her hidden layer'da uygulandı
- Validation accuracy'de iyileşme sağladı

### S18: Sequence length neden 60 frame?
**Cevap:**
- Video'ların ortalama uzunluğuna göre seçildi
- Çok kısa: Yeterli bilgi yok
- Çok uzun: Hesaplama maliyeti artar
- 60 frame ≈ 2-3 saniye (30 FPS'de)
- Deneysel olarak optimal bulundu

---

## 🎓 EĞİTİM SÜRECİ HAKKINDA SORULAR

### S19: Eğitim parametreleriniz neler?
**Cevap:**
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Epoch**: 50
- **Batch Size**: 32 (image), 16 (sequence)
- **Validation Ratio**: 0.2 (%20)

### S20: Neden Adam optimizer seçtiniz?
**Cevap:**
- Adaptive learning rate (her parametre için ayrı)
- Momentum ve RMSprop'un birleşimi
- Hızlı yakınsama
- Standart ve etkili
- PyTorch'ta kolay kullanım

### S21: Learning rate scheduler neden kullandınız?
**Cevap:**
- Validation loss'a göre otomatik ayarlama
- Plateau'da takılmayı önler
- Daha iyi sonuçlara ulaşma
- Patience=5: 5 epoch bekle, iyileşme yoksa lr'yi yarıya indir
- Eğitim sürecini optimize eder

### S22: Eğitim ne kadar sürdü?
**Cevap:**
- Image Model: ~2-3 saat (CPU'da)
- Sequence Model: ~3-4 saat (CPU'da)
- GPU kullanılsaydı daha hızlı olurdu
- Toplam: ~5-7 saat (her iki model)
- 50 epoch × her epoch ~5-10 dakika

### S23: Overfitting problemi yaşadınız mı?
**Cevap:**
- Hayır, Dropout ile önlendi
- Train ve validation loss birlikte azaldı
- Validation accuracy düzenli arttı
- Early stopping gerekmedi
- Model genelleştirme yeteneği iyi

---

## 📈 DEĞERLENDİRME HAKKINDA SORULAR

### S24: Model performansınız nasıl?
**Cevap:**
- **Image Model**: Validation accuracy ~91%
- **Sequence Model**: Validation accuracy ~99%
- Sequence Model daha başarılı (zaman serisi bilgisi)
- Her iki model de başarıyla eğitildi
- Confusion matrix ile detaylı analiz yapıldı

### S25: Hangi metrikleri kullandınız?
**Cevap:**
- **Accuracy**: Genel doğruluk oranı
- **Precision**: Pozitif tahminlerin doğruluğu
- **Recall**: Gerçek pozitiflerin tespit oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması
- **Confusion Matrix**: Hangi sınıfların karıştırıldığını gösterir

### S26: Hangi egzersizler daha zor tanındı?
**Cevap:**
- Benzer hareketler karıştırılabiliyor
- Örnek: bench press vs incline bench press
- Confusion matrix'te görülebilir
- Daha fazla eğitim verisi ile iyileştirilebilir
- Model genel olarak iyi performans gösterdi

---

## 🔄 GERÇEK KULLANIM HAKKINDA SORULAR

### S27: Kullanıcı video yüklediğinde sistem nasıl çalışıyor?
**Cevap:**
1. Video geçici olarak kaydedilir
2. `pose_detector.process_video()` çağrılır
3. Video frame frame okunur (OpenCV)
4. Her frame'de MediaPipe ile pose detection
5. Her frame'den keypoints çıkarılır (132 boyut)
6. Keypoints array'i oluşturulur: (frame_count, 132)
7. Sequence length'e göre hazırlanır (60 frame)
8. Sequence Model'e verilir
9. Tahmin yapılır ve sonuç döndürülür

### S28: Frame'ler görsel dosyası olarak kaydediliyor mu?
**Cevap:**
- **Hayır**, sadece memory'de işleniyor
- Görsel dosyası olarak kaydedilmiyor
- Her frame'den direkt keypoints çıkarılıyor
- Daha hızlı ve verimli
- Disk kullanımı azalır

### S29: Gerçek zamanlı işleme yapılabiliyor mu?
**Cevap:**
- Şu an batch işleme (video yükle → işle → sonuç)
- Gerçek zamanlı için kamera feed'i gerekir
- MediaPipe gerçek zamanlı destekler
- Gelecek geliştirme olarak eklenebilir
- Webcam entegrasyonu yapılabilir

---

## 🚀 GELECEK GELİŞTİRMELER HAKKINDA SORULAR

### S30: Projeyi nasıl geliştirebilirsiniz?
**Cevap:**
- **Veri arttırma**: Rotation, scale, noise ekleme
- **Ek özellikler**: Açı hesaplama, mesafe, hız
- **Form analizi**: Doğru/yanlış form tespiti
- **Geri bildirim**: Kullanıcıya öneriler
- **Hibrit model**: Image + Sequence birleştirme
- **Gerçek zamanlı**: Webcam entegrasyonu
- **Daha fazla egzersiz**: 22'den daha fazla sınıf

### S31: Form analizi yapıyor musunuz?
**Cevap:**
- Şu an sadece egzersiz tanıma yapılıyor
- Form analizi gelecek geliştirme
- Keypoints'lerden açı hesaplama yapılabilir
- Doğru/yanlış form karşılaştırması
- Kullanıcıya geri bildirim sistemi

### S32: Modeli production'a nasıl alırsınız?
**Cevap:**
- Model optimizasyonu (quantization)
- API servisi (FastAPI/Flask)
- Cloud deployment (AWS, GCP, Azure)
- Docker containerization
- Caching ve load balancing
- Monitoring ve logging

---

## 🐛 TEKNİK SORUNLAR HAKKINDA SORULAR

### S33: Windows'ta MediaPipe ile sorun yaşadınız mı?
**Cevap:**
- Evet, Türkçe karakter sorunları
- Short path kullanarak çözüldü
- GPU desteği kapatıldı (CPU modu)
- Static mode kullanıldı (video için)
- Çalışma zamanında düzeltildi

### S34: Veri leakage problemi yaşadınız mı?
**Cevap:**
- Hayır, dikkatli train/test split yapıldı
- Test videolarından görsel çıkarılmadı
- Aynı videodan hem görsel hem video kullanılmadı (test'te)
- Random seed ile tekrarlanabilir split
- Validation seti train'den ayrıldı

### S35: Model eğitimi sırasında hata aldınız mı?
**Cevap:**
- İlk başta sınıf sayısı uyumsuzluğu
- Görsel ve video sınıfları eşleştirme sorunu
- Label encoder uyumluluğu çözüldü
- Ortak sınıflar filtrelendi
- Kodda düzeltmeler yapıldı

---

## 📚 AKADEMİK/ARAŞTIRMA SORULARI

### S36: Literatürde benzer çalışmalar var mı?
**Cevap:**
- Evet, egzersiz tanıma alanında çalışmalar mevcut
- MediaPipe kullanan çalışmalar var
- LSTM ile video sınıflandırma yaygın
- Bizim yaklaşımımız: Hibrit (görsel + video)
- 22 egzersiz sınıfı ile kapsamlı

### S37: Projenin bilimsel katkısı nedir?
**Cevap:**
- Hibrit yaklaşım (görsel + video)
- MediaPipe keypoints ile egzersiz tanıma
- Pratik uygulama (kişisel antrenör)
- Açık kaynak kod ve veri seti
- Tekrarlanabilir metodoloji

### S38: Hangi makine öğrenmesi tekniklerini kullandınız?
**Cevap:**
- **Supervised Learning**: Etiketli veri ile eğitim
- **Deep Learning**: MLP ve LSTM
- **Transfer Learning**: MediaPipe pre-trained model
- **Time Series Analysis**: LSTM ile sequence analizi
- **Classification**: Çok sınıflı sınıflandırma

---

## 💡 PRATİK SORULAR

### S39: Projeyi çalıştırmak için ne gerekli?
**Cevap:**
- Python 3.9+
- PyTorch 2.4.0 (CUDA 12.1 desteği)
- MediaPipe 0.10.9
- OpenCV, NumPy, scikit-learn
- Eğitilmiş modeller (`models/` klasörü)
- İşlenmiş veri (`data/processed/`)

### S40: Kodunuz açık kaynak mı?
**Cevap:**
- Evet, GitHub'da paylaşılabilir
- Açık kaynak lisansı (LICENSE dosyası var)
- Başkaları kullanabilir ve geliştirebilir
- Akademik kullanım için uygun

---

## 🎯 ÖZET CEVAPLAR (Hızlı Referans)

**Proje Amacı**: 22 egzersiz türünü video/görsellerden otomatik tanıma

**Teknolojiler**: PyTorch, MediaPipe, OpenCV, LSTM, MLP

**Veri**: Kaggle'dan açık kaynak, 22 sınıf, %70 train / %30 test

**Modeller**: Image Classifier (MLP) + Sequence Classifier (LSTM)

**Performans**: Image ~91%, Sequence ~99% accuracy

**Gerçek Kullanım**: Video yükle → Frame frame işle → Keypoints çıkar → Model tahmin

**Gelecek**: Form analizi, gerçek zamanlı, daha fazla egzersiz

---

## 💬 SUNUM İÇİN İPUÇLARI

1. **Güvenli konuşun**: Projeyi siz yaptınız, detayları biliyorsunuz
2. **Açık olun**: Bilmediğiniz bir şey varsa "Bu konuda daha fazla araştırma yapabilirim" deyin
3. **Örnekler verin**: Kod örnekleri, grafikler, sonuçlar gösterin
4. **Zorlukları anlatın**: Karşılaştığınız sorunlar ve çözümler
5. **Gelecek planlarınızı belirtin**: Projeyi nasıl geliştireceğiniz

---

**Başarılar! 🚀**

