import streamlit as st
import requests
import os
from pathlib import Path

# Sayfa yapılandırması
st.set_page_config(page_title="Egzersiz Sınıflandırma Uygulaması", layout="wide")

# Başlık
st.title("🤸‍♂️ Egzersiz Sınıflandırma ve Form Analizi")
st.markdown("Yüklediğiniz videodan egzersiz türünü ve formunu analiz edin.")

# Backend API URL'si
# Eğer backend farklı bir makinede çalışıyorsa bu URL'yi güncelleyin
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

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

