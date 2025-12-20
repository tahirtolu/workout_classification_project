# Bu dosya artık kullanılmıyor.
#
# Yeni uygulama iki parçaya ayrılmıştır:
# 1. Backend API: `backend/backend.py` (FastAPI)
# 2. Frontend Uygulaması: `frontend/frontend.py` (Streamlit)
#
# Nasıl Çalıştırılır:
# 1. Backend'i başlatın (tercihen farklı bir terminalde):
#    cd backend
#    uvicorn backend:app --reload --host 0.0.0.0 --port 8000
#
# 2. Frontend'i başlatın (ayrı bir terminalde):
#    streamlit run frontend/frontend.py --server.port 8503
#
# Sanal ortamın etkin olduğundan ve tüm bağımlılıkların yüklü olduğundan emin olun.
