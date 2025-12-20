import os
import sys
import cv2
import numpy as np
import json
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

# Proje kök dizinini Python yoluna ekle
project_root = Path(__file__).resolve().parents[2] # backend/main.py -> backend -> project_root
sys.path.insert(0, str(project_root))

from src.pose_detector import PoseDetector
from src.models.exercise_classifier import SequenceClassifier

app = FastAPI(
    title="Workout Classification API",
    description="Bu API, yüklenen videolardan egzersiz türünü ve formunu analiz eder.",
    version="1.0.0"
)

# CORS ayarları
origins = [
    "http://localhost",
    "http://localhost:3000", # React uygulamaları için
    "http://localhost:8503", # Streamlit uygulaması için
    # Diğer frontend URL'lerinizi buraya ekleyin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
pose_detector_instance: PoseDetector = None
model: SequenceClassifier = None
class_names: list = None

MODEL_PATH = project_root / "models" / "sequence_classifier" / "best_model.pth"
CLASS_NAMES_PATH = project_root / "models" / "sequence_classifier" / "class_names.json"
UPLOAD_DIR = project_root / "uploaded_videos"

# Uygulama başlangıcında modelleri yükle
@app.on_event("startup")
async def startup_event():
    global pose_detector_instance, model, class_names
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    print("Backend başlatılıyor...")

    # Pose Detector yükle
    pose_detector_instance = PoseDetector()

    # Sınıf isimlerini yükle
    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(f"Hata: Sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    
    num_classes = len(class_names)
    
    # Modeli yükle
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Hata: Model dosyası bulunamadı: {MODEL_PATH}")

    model = SequenceClassifier(num_classes=num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Backend hazır!")

@app.post("/classify-video/")
async def classify_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Sadece video dosyalari yuklenebilir (.mp4, .mov, .avi).")

    temp_video_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_video_path, "wb+") as file_object:
            file_object.write(file.file.read())

        # 1. Keypoint çıkarma
        video_keypoints = pose_detector_instance.process_video(str(temp_video_path), display=False, verbose=False)
        if video_keypoints is None or len(video_keypoints) == 0:
            raise HTTPException(status_code=400, detail="Videodan anahtar nokta çıkarılamadı veya hiç poz algılanmadı.")

        sequence_length = 60
        processed_keypoints = torch.tensor(video_keypoints, dtype=torch.float32).unsqueeze(0)

        if processed_keypoints.shape[1] > sequence_length:
            processed_keypoints = processed_keypoints[:, -sequence_length:, :]
        elif processed_keypoints.shape[1] < sequence_length:
            padding_needed = sequence_length - processed_keypoints.shape[1]
            padding = torch.zeros(1, padding_needed, 132, dtype=torch.float32)
            processed_keypoints = torch.cat((processed_keypoints, padding), dim=1)

        # 3. Sınıflandırma yapma
        with torch.no_grad():
            predicted_class_idx, probabilities = model.predict(processed_keypoints)
            predicted_class_name = class_names[predicted_class_idx]
            
            # En yüksek 3 olasılığı al
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            top_3_predictions = [{
                "exercise": class_names[idx],
                "probability": float(probabilities[idx]*100)
            } for idx in top_3_indices]

        return JSONResponse(content={
            "message": "Video basariyla siniflandirildi.",
            "prediction": {
                "exercise": predicted_class_name,
                "probability": float(probabilities[predicted_class_idx]*100),
                "top_predictions": top_3_predictions
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata olustu: {e}")
    finally:
        if temp_video_path.exists():
            os.remove(temp_video_path)


@app.get("/")
async def read_root():
    return {"message": "Workout Classification API'a hos geldiniz"}

if __name__ == "__main__":
    # Kök dizinden çalıştırıldığında (uvicorn backend.backend:app --reload)
    # Uvicorn'ı başlat (varsayılan port 8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
