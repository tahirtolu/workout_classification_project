import streamlit as st
import requests
import os
import subprocess
import time
import sys
import cv2
import numpy as np
import json
import torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Proje kök dizinini Python yoluna ekle
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pose_detector import PoseDetector
from src.models.exercise_classifier import SequenceClassifier

# FastAPI backend URL'si
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# Sayfa yapılandırması
st.set_page_config(page_title="Egzersiz Sınıflandırma Uygulaması", layout="wide")

# Başlık
st.title("🤸‍♂️ Egzersiz Sınıflandırma ve Form Analizi")
st.markdown("Yüklediğiniz videodan egzersiz türünü ve formunu analiz edin.")

# Backend'i başlatmak için fonksiyon
def start_backend():
    # Sanal ortamın python.exe yolunu bul
    venv_python_path = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python_path):
        st.error(f"Hata: Sanal ortamdaki Python bulunamadi: {venv_python_path}")
        st.stop()

    # Uvicorn komutu
    cmd = [venv_python_path, "-m", "uvicorn", "main:app", "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)]

    # Backend'i arka planda başlat
    process = subprocess.Popen(cmd, cwd=project_root) # Kök dizinden çalıştır
    st.session_state["backend_process"] = process
    time.sleep(5) # Backend'in başlaması için biraz bekle

# Backend FastAPI uygulaması
app = FastAPI(
    title="Workout Classification API",
    description="Bu API, yüklenen videolardan egzersiz türünü ve formunu analiz eder.",
    version="1.0.0"
)

# CORS ayarları
origins = [
    "http://localhost",
    "http://localhost:8503", # Streamlit uygulaması için
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

# Uygulama başlangıcında modelleri yükle (FastAPI için)
@app.on_event("startup")
async def startup_event_backend():
    global pose_detector_instance, model, class_names
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    st.session_state.backend_status = "Başlatılıyor..."
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
    
    st.session_state.backend_status = "Hazır!"
    print("Backend hazır!")

@app.post("/classify-video/")
async def classify_video_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Sadece video dosyalari yuklenebilir (.mp4, .mov, .avi).")

    temp_video_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_video_path, "wb+") as file_object:
            file_object.write(await file.read())

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
async def read_root_backend():
    return {"message": "Workout Classification API'a hos geldiniz"}

# --- Streamlit Arayüzü --- #

# Backend'i başlat
if "backend_process" not in st.session_state:
    start_backend()


st.subheader("Video Yükle")
uploaded_file = st.file_uploader("Bir egzersiz videosu yükleyin (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Analizi Başlat"):
        st.subheader("Analiz Sonuçları")
        
        with st.spinner("Video analizi başlatılıyor..."):
            try:
                # Backend API'sine video gönder
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/classify-video/", files=files, timeout=300) # 5 dakika timeout
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Analiz başarıyla tamamlandı!")
                    
                    prediction_data = result.get("prediction", {})
                    exercise = prediction_data.get("exercise")
                    probability = prediction_data.get("probability")
                    top_predictions = prediction_data.get("top_predictions", [])

                    if exercise and probability:
                        st.write(f"**Tahmin Edilen Egzersiz:** {exercise}")
                        st.write(f"**Olasılık:** {probability:.2f}%")
                        
                        if top_predictions:
                            st.markdown("---")
                            st.subheader("En Yüksek Olasılıklar")
                            for i, pred in enumerate(top_predictions):
                                st.write(f"{i+1}. {pred['exercise']}: {pred['probability']:.2f}%")
                    else:
                        st.warning("Tahmin verileri alınamadı.")

                elif response.status_code == 400:
                    error_detail = response.json().get("detail", "Bilinmeyen bir hata oluştu.")
                    st.error(f"Analiz hatası: {error_detail}")
                else:
                    st.error(f"Backend sunucusundan hata kodu alındı: {response.status_code} - {response.text}")
            except requests.exceptions.Timeout:
                st.error("Analiz zaman aşımına uğradı. Lütfen daha kısa bir video deneyin veya internet bağlantınızı kontrol edin.")
            except requests.exceptions.ConnectionError:
                st.error(f"Backend sunucusuna bağlanılamadı. Lütfen backend'in {BACKEND_URL} adresinde çalıştığından emin olun.")
            except Exception as e:
                st.error(f"Beklenmeyen bir hata oluştu: {e}")
else:
    st.info("Lütfen bir video yükleyin.")

# Streamlit uygulaması kapatıldığında backend prosesini sonlandır
@st.cache_data(show_spinner=False, ttl=300)
def cleanup_backend_on_exit():
    # Streamlit tekrar çalıştırıldığında veya kapatıldığında çağrılır
    if "backend_process" in st.session_state and st.session_state["backend_process"].poll() is None:
        st.session_state["backend_process"].terminate()
        st.session_state["backend_process"].wait(timeout=5) # Kapatmayı bekle
        print("Backend prosesi sonlandirildi.")

cleanup_backend_on_exit()