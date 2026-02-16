# 🧠 Medical Risk MLOps — Stroke Risk Prediction

CatBoost + MLflow + FastAPI + Streamlit ile uçtan uca **inme (stroke) riski tahmini** sistemi.

---

## 📋 İçindekiler

- [Proje Yapısı](#-proje-yapısı)
- [Ne Yaptık ve Neden?](#-ne-yaptık-ve-neden)
- [Kurulum](#-kurulum)
- [Çalıştırma Adımları](#-çalıştırma-adımları)
- [Docker ile Çalıştırma](#-docker-ile-çalıştırma)
- [Streamlit Cloud'a Deploy](#-streamlit-clouda-deploy)
- [API Kullanımı](#-api-kullanımı)
- [Teknolojiler](#-teknolojiler)

---

## 🗂 Proje Yapısı

```
medical-risk-mlops/
│
├── data/
│   └── raw/stroke_data.csv          # Kaggle Stroke Prediction veri seti (5110 kayıt)
│
├── models/
│   ├── model.cbm                    # Standalone CatBoost model (Docker/Cloud için)
│   └── metadata.json                # Threshold ve model bilgileri
│
├── src/
│   ├── config.py                    # Tüm ayarlar: yollar, hiperparametreler, feature listesi
│   │
│   ├── data/
│   │   ├── load_data.py             # CSV okuma
│   │   ├── validate.py              # Veri doğrulama (eksik değer, dtype kontrol)
│   │   └── preprocess.py            # Temizleme, BMI doldurma, train/test split
│   │
│   ├── models/
│   │   ├── train.py                 # Model eğitimi + MLflow loglama + SHAP
│   │   ├── evaluate.py              # Metrikler + threshold tuning
│   │   └── predict.py               # MLflow'dan model yükleme + tahmin
│   │
│   └── api/
│       ├── schema.py                # Pydantic giriş/çıkış şemaları
│       └── main.py                  # FastAPI endpoint'leri (/predict, /explain, /health)
│
├── .streamlit/config.toml           # Streamlit tema ve server ayarları
├── streamlit_app.py                 # 🖥 Streamlit dashboard (görsel arayüz)
├── Dockerfile                       # Multi-stage Docker container
├── docker-compose.yml               # 3 servis: API + Streamlit + MLflow UI
├── requirements.txt                 # Python bağımlılıkları
└── README.md                        # Bu dosya
```

---

## 🎯 Ne Yaptık ve Neden?

### 1. `config.py` — Merkezi Ayar Dosyası

**Neden:** Hiperparametreler, dosya yolları ve feature isimleri tek yerde olsun ki her dosyada tekrar yazılmasın. Bir şeyi değiştirmek istersen sadece burayı değiştirirsin.

### 2. `load_data.py` → `validate.py` → `preprocess.py` — Veri Hattı

**Neden:** Veri yükleme → doğrulama → temizleme adımlarını ayrı modüllere böldük. Her biri bağımsız olarak test edilebilir ve değiştirilebilir.

| Adım            | Ne Yapar                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------ |
| `load_data.py`  | CSV dosyasını okur                                                                               |
| `validate.py`   | Eksik değerleri, target dağılımını ve dtype'ları kontrol eder                                    |
| `preprocess.py` | `id` sütununu düşürür, BMI'deki null'ları medyan ile doldurur, stratified train/test split yapar |

### 3. `train.py` — Model Eğitimi + MLflow

**Neden:** CatBoost modeli eğitir ve her şeyi MLflow'a kaydeder → tekrarlanabilirlik sağlar.

**Ne loglar:**

- Hiperparametreler (iterations, depth, learning_rate, class_weights)
- Metrikler: ROC-AUC, Precision, Recall, F1, optimal threshold
- Artifactler: confusion matrix (PNG + JSON), classification report (TXT), SHAP özet grafiği (PNG)
- Modelin kendisi → MLflow Model Registry'ye kaydeder
- Standalone export → `models/model.cbm` + `models/metadata.json` (Docker/Cloud için)

### 4. `evaluate.py` — Threshold Tuning

**Neden:** Veri setinde **%95 no-stroke** vs **%5 stroke** var (aşırı dengesiz). Varsayılan 0.5 threshold çok fazla stroke vakasını kaçırır. **F2-score** ile recall'u optimize eden optimal threshold buluruz (≈0.69).

### 5. `predict.py` — Model Yükleme ve Tahmin

**Neden:** MLflow Registry'den modeli yükler ve tek bir hasta verisi için tahmin yapar. API ve Streamlit bu modülü kullanır.

### 6. `schema.py` — Pydantic Şemaları

**Neden:** API'ye gelen verilerin doğruluğunu garanti eder. Yanlış tip veya eksik alan gönderirsen hata mesajı döner.

### 7. `api/main.py` — FastAPI REST API

**Neden:** Modeli bir HTTP servisi olarak sunar. Herhangi bir uygulama (web, mobil, başka servis) bu API'yi çağırarak tahmin alabilir.

| Endpoint   | Method | Açıklama                             |
| ---------- | ------ | ------------------------------------ |
| `/health`  | GET    | Sistem durumu kontrolü               |
| `/predict` | POST   | Tek hasta için stroke risk tahmini   |
| `/explain` | POST   | SHAP değerleri ile tahmin açıklaması |

### 8. `streamlit_app.py` — Dashboard Arayüzü

**Neden:** Teknik olmayan kullanıcılar için görsel arayüz. Hasta bilgilerini doldur → tahmin al → SHAP grafiğiyle hangi faktörlerin riski artırdığını gör.

### 9. Dual-Mode Model Loading

**Neden:** Eğitim sonrası model iki yere kaydedilir:

1. **MLflow Registry** → Local development için (deney takibi ile birlikte)
2. **`models/model.cbm`** → Docker ve Cloud deploy için (MLflow bağımlılığı yok)

API ve Streamlit önce `.cbm` dosyasını arar → bulamazsa MLflow'a düşer.

### 10. Class Imbalance Çözümü

**Neden:** 4861 no-stroke vs 249 stroke. `class_weights=[1, 20]` ile CatBoost'a stroke vakalarını 20x daha önemli olarak öğretiyoruz.

---

## ⚙️ Kurulum

```bash
# 1. Sanal ortam oluştur
python -m venv .venv

# 2. Sanal ortamı aktifle
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## 🚀 Çalıştırma Adımları

### Adım 1 — Model Eğitimi

```bash
python -m src.models.train
```

**Ne olur:**

- Veri yüklenir ve işlenir
- CatBoost modeli eğitilir (500 iterasyon)
- Threshold optimize edilir (recall için)
- SHAP grafiği oluşturulur
- Her şey MLflow'a loglanır
- Model MLflow Registry'ye kaydedilir
- `models/model.cbm` ve `models/metadata.json` oluşturulur

**Çıktı:** `ROC-AUC: ~0.85 | Recall: ~0.74 | Threshold: ~0.69`

### Adım 2a — Streamlit Dashboard (Önerilen)

```bash
streamlit run streamlit_app.py --server.port 8890
```

Tarayıcıda **http://localhost:8890** adresini aç.

> ⚠️ **Windows Hyper-V Notu:** Port 8501 (varsayılan) Hyper-V tarafından bloke olabilir. `--server.port 8890` ekleyerek farklı bir port kullan.

### Adım 2b — FastAPI (Alternatif)

```bash
uvicorn src.api.main:app --port 8000
```

API: **http://localhost:8000/docs** (Swagger UI)

### Adım 3 — MLflow UI (Opsiyonel)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**http://localhost:5000** → Tüm deneyleri, metrikleri ve artifactleri görsel olarak incele.

---

## 🐳 Docker ile Çalıştırma

### Tek Servis

```bash
# Image oluştur
docker build -t stroke-risk-mlops .

# FastAPI çalıştır
docker run -p 8000:8000 stroke-risk-mlops uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Streamlit çalıştır
docker run -p 8501:8501 stroke-risk-mlops streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Compose (3 Servis Birden)

```bash
docker-compose up -d
```

| Servis        | URL                   | Açıklama              |
| ------------- | --------------------- | --------------------- |
| **API**       | http://localhost:8000 | FastAPI REST endpoint |
| **Streamlit** | http://localhost:8501 | Dashboard arayüzü     |
| **MLflow**    | http://localhost:5000 | Deney takip arayüzü   |

```bash
# Durdur
docker-compose down
```

> **Not:** Docker container'ları `models/model.cbm` dosyasını kullanır (MLflow registry'ye bağımlı değildir). Bu sayede Windows'ta eğitilen model Linux container'da sorunsuz çalışır.

---

## ☁️ Streamlit Cloud'a Deploy

1. Projeyi **GitHub'a push** et
2. [share.streamlit.io](https://share.streamlit.io) adresine git
3. GitHub reposunu seç → `streamlit_app.py` dosyasını seç
4. **Deploy** tıkla

> **Önemli:** `models/model.cbm` ve `models/metadata.json` dosyalarının repo'da olduğundan emin ol (`.gitignore`'da olmamalı).

---

## 📡 API Kullanımı

### Tahmin İsteği

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

### Yanıt

```json
{
  "prediction": 1,
  "probability_stroke": 0.8357
}
```

---

## 🛠 Teknolojiler

| Teknoloji        | Kullanım Amacı                                          |
| ---------------- | ------------------------------------------------------- |
| **CatBoost**     | Kategorik veri desteği olan gradient boosting modeli    |
| **MLflow**       | Deney takibi, model kayıt, artifact saklama             |
| **FastAPI**      | REST API (yüksek performanslı, otomatik dokümantasyon)  |
| **Streamlit**    | İnteraktif dashboard arayüzü                            |
| **SHAP**         | Model açıklanabilirliği (hangi feature ne kadar etkili) |
| **Pydantic**     | Veri doğrulama (API giriş/çıkış)                        |
| **Docker**       | Container ile taşınabilir dağıtım                       |
| **scikit-learn** | Train/test split, metrik hesaplama                      |

---

## 📊 Model Performansı

| Metrik        | Değer      |
| ------------- | ---------- |
| ROC-AUC       | **0.8485** |
| Recall        | **0.7400** |
| Threshold     | **0.6904** |
| Class Weights | [1, 20]    |

---

_Built with ❤️ using CatBoost + MLflow + FastAPI + Streamlit_
