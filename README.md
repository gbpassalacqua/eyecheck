# Guard2Live V2.1 — API Server

TFLite inference server para screening ocular com 9 classes.

## Deploy Rápido (Render.com — GRÁTIS)

### 1. Crie conta em render.com

### 2. Suba este projeto pro GitHub
```bash
git init
git add .
git commit -m "Guard2Live API"
git remote add origin https://github.com/SEU_USER/guard2live-api.git
git push -u origin main
```

### 3. No Render Dashboard:
- **New** → **Web Service**
- Conecte seu GitHub repo
- **Environment**: Docker
- **Plan**: Free
- **Deploy**

### 4. Sua API vai estar em:
```
https://guard2live-api.onrender.com
```

### 5. Teste:
```bash
curl https://guard2live-api.onrender.com/health
# {"status": "ok"}
```

## API Endpoints

### GET /
Status do servidor

### GET /health
Health check

### POST /analyze
Upload de arquivo
```bash
curl -X POST -F "file=@eye_photo.jpg" https://guard2live-api.onrender.com/analyze
```

### POST /analyze-base64
Imagem em base64 (usado pelo React demo)
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/..."}' \
  https://guard2live-api.onrender.com/analyze-base64
```

### Response
```json
{
  "classification": "Conjunctivitis",
  "confidence": 0.7234,
  "scores": {
    "Normal": 0.1, "Cataract": 0.02, "Conjunctivitis": 0.72,
    "Hemorrhage": 0.05, "Jaundice": 0.01, "Pterygium": 0.02,
    "Ptosis": 0.03, "Stye/Chalazion": 0.01, "Uveitis": 0.04
  },
  "findings": ["Conjunctival redness detected", ...],
  "systemic_alert": null,
  "urgency": "low",
  "recommendation": "Consult an ophthalmologist...",
  "model": "MobileNetV2-TFLite"
}
```

## Alternativa: Railway.app
```bash
railway login
railway init
railway up
```

## Alternativa: Local
```bash
pip install -r requirements.txt
python server.py
# Server runs on http://localhost:8000
```

## Modelo
- **Arquitetura**: MobileNetV2 (transfer learning do ImageNet)
- **Dataset**: 1,157 imagens curadas, 9 classes
- **AUC médio**: 0.862
- **TFLite size**: 5.8 MB
- **Input**: 224x224 RGB

## Classes
1. Normal ✅
2. Cataract ⚪
3. Conjunctivitis 🔴
4. Hemorrhage 🩸 (sistêmica: hipertensão)
5. Jaundice 🟡 (sistêmica: fígado)
6. Pterygium 🔺
7. Ptosis 👁️ (sistêmica: AVC)
8. Stye/Chalazion 🫧
9. Uveitis 🟠 (sistêmica: autoimune)

---
**BBG © 2026 — MIT xPRO Project**
