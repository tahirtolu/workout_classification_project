"""
Temel pose detection modülü - MediaPipe kullanarak vücut pozlarını tespit eder
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# Windows'ta Türkçe karakter sorununu çözmek için MediaPipe'ı patch et
if sys.platform == 'win32':
    try:
        import win32api
        # MediaPipe'ın model dosyası yollarını düzelt
        original_open = open
        
        def patched_open(file, mode='r', *args, **kwargs):
            # Dosya yolunu short path'e çevir
            if isinstance(file, str) and os.path.exists(file):
                try:
                    file = win32api.GetShortPathName(file)
                except:
                    pass
            return original_open(file, mode, *args, **kwargs)
        
        # builtins.open'ı patch et (MediaPipe kullanır)
        import builtins
        builtins.open = patched_open
    except ImportError:
        pass  # pywin32 yoksa normal devam et


class PoseDetector:
    """MediaPipe kullanarak vücut pozisyon tespiti yapan sınıf"""
    
    def __init__(self):
        # Windows'ta Türkçe karakter sorununu çözmek için short path kullan
        if sys.platform == 'win32':
            try:
                import win32api
                # MediaPipe modülünün yolunu short path'e çevir
                mediapipe_dir = os.path.dirname(mp.__file__)
                short_path = win32api.GetShortPathName(mediapipe_dir)
                # MediaPipe için GPU devre dışı (daha stabil)
                try:
                    import torch
                    if torch.cuda.is_available():
                        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
                        print("ℹ️  PyTorch GPU kullanabilir, MediaPipe CPU modunda kalacak")
                    else:
                        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
                except ImportError:
                    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
            except ImportError:
                pass  # pywin32 yoksa normal devam et
            except Exception as e:
                pass  # Hata varsa devam et
        
        self.mp_pose = mp.solutions.pose
        # Model dosyasının varlığını kontrol et
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Daha düşük karmaşıklık seviyesi
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except (FileNotFoundError, RuntimeError) as e:
            # Eğer hala hata varsa, static_image_mode kullan
            print(f"UYARI: Video modu basarisiz, static mod deneniyor...")
            print(f"Hata detayi: {e}")
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,  # Video için her frame'i ayrı işle
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def process_frame(self, frame):
        """Tek bir frame'i işler ve pose landmarks'larını döndürür"""
        # BGR'den RGB'ye çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Static mode kullanılıyorsa, her frame'i ayrı işle
        # Video mode'da otomatik tracking var
        results = self.pose.process(rgb_frame)
        return results
    
    def draw_pose(self, frame, results):
        """Frame üzerine pose landmarks'larını çizer"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def extract_keypoints(self, results):
        """Pose landmarks'larından keypoint array'i çıkarır"""
        if not results.pose_landmarks:
            return None
        
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return np.array(keypoints)
    
    def process_video(self, video_path, output_path=None, display=True, verbose=False):
        """
        Videoyu işler ve pose detection yapar
        
        Args:
            video_path: Video dosya yolu
            output_path: Çıktı video yolu (opsiyonel)
            display: Görüntüyü göster (varsayılan: True)
            verbose: Detaylı çıktı göster (varsayılan: False, data_collector için sessiz)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        # Video özelliklerini al
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sadece verbose=True ise detaylı bilgi göster
        if verbose:
            print(f"Video bilgileri:")
            print(f"  Boyut: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  Toplam frame: {total_frames}")
            print(f"  Süre: {total_frames/fps:.2f} saniye")
            print("\nVideo işleniyor...")
        
        # Output video writer (eğer output_path verilmişse)
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_keypoints = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Pose detection
            results = self.process_frame(frame)
            
            # Keypoints çıkar
            keypoints = self.extract_keypoints(results)
            if keypoints is not None:
                all_keypoints.append(keypoints)
            
            # Pose çiz
            frame = self.draw_pose(frame, results)
            
            # Frame numarası ekle
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Pose tespit edildi mi bilgisi
            status = "Pose detected" if results.pose_landmarks else "No pose"
            cv2.putText(frame, status, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if results.pose_landmarks else (0, 0, 255), 2)
            
            # Output'a yaz
            if out:
                out.write(frame)
            
            # Göster (eğer display True ise)
            if display:
                cv2.imshow('Pose Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if verbose:
                        print("\nKullanıcı tarafından durduruldu (q tuşu)")
                    break
            
            frame_count += 1
        
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        # Sadece verbose=True ise detaylı bilgi göster
        if verbose:
            print(f"\nİşlem tamamlandı!")
            print(f"  İşlenen frame: {frame_count}")
            print(f"  Pose tespit edilen frame: {len(all_keypoints)}")
            if frame_count > 0:
                print(f"  Başarı oranı: {len(all_keypoints)/frame_count*100:.2f}%")
        
        return np.array(all_keypoints) if all_keypoints else None
    
    def release(self):
        """Kaynakları serbest bırak"""
        self.pose.close()

