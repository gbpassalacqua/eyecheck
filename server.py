"""
EYECHECK V2.1 — TFLite Inference API
Deploy on Render, Railway, or any Python host
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
from PIL import Image
import io
import os

# Use tflite-runtime / ai-edge-litert (lightweight) or fall back to full tensorflow
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite-runtime")
except ImportError:
    try:
        from ai_edge_litert import interpreter as tflite
        print("Using ai-edge-litert")
    except ImportError:
        import tensorflow.lite as tflite
        print("Using tensorflow.lite")

app = FastAPI(title="EYECHECK V2.1 API", version="2.1.0")

# CORS — allow all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class metadata
CLASSES = [
    {"name": "Cataract",       "systemic": None,                                        "urgency": "moderate"},
    {"name": "Conjunctivitis", "systemic": None,                                        "urgency": "low"},
    {"name": "Hemorrhage",     "systemic": "Hipertensão arterial, distúrbios sanguíneos","urgency": "moderate"},
    {"name": "Jaundice",       "systemic": "Disfunção hepática, hepatite, obstrução biliar", "urgency": "high"},
    {"name": "Normal",         "systemic": None,                                        "urgency": "none"},
    {"name": "Pterygium",      "systemic": None,                                        "urgency": "low"},
    {"name": "Ptosis",         "systemic": "AVC, miastenia gravis, dano nervoso",       "urgency": "high"},
    {"name": "Stye/Chalazion", "systemic": None,                                        "urgency": "low"},
    {"name": "Uveitis",        "systemic": "Doenças autoimunes, infecções sistêmicas",  "urgency": "moderate"},
]

# Load TFLite model
MODEL_PATH = os.environ.get("MODEL_PATH", "guard2live_v2_9class.tflite")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SIZE = input_details[0]['shape'][1]  # 224

print(f"Model loaded: {MODEL_PATH}")
print(f"Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
print(f"Output: {output_details[0]['shape']}")
print(f"Classes: {[c['name'] for c in CLASSES]}")


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Resize and normalize image for MobileNetV2"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocessing: scale to [-1, 1]
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence to report a condition


def run_inference(image_bytes: bytes) -> dict:
    """Run TFLite inference and return structured result"""
    tensor = preprocess(image_bytes)
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Build scores dict
    scores = {}
    for i, cls in enumerate(CLASSES):
        scores[cls["name"]] = round(float(output[i]), 4)

    # Top prediction
    top_idx = int(np.argmax(output))
    top_class = CLASSES[top_idx]
    confidence = float(output[top_idx])

    # If confidence is below threshold and it's NOT already Normal,
    # override to Normal — low confidence should not alarm the user
    if confidence < CONFIDENCE_THRESHOLD and top_class["name"] != "Normal":
        # Find Normal class index
        normal_idx = next(i for i, c in enumerate(CLASSES) if c["name"] == "Normal")
        normal_class = CLASSES[normal_idx]
        normal_score = float(output[normal_idx])

        findings = [
            "Nenhuma condição detectada com confiança suficiente",
            "Aparência geral dentro da normalidade",
            f"Maior suspeita: {CLASS_PT.get(top_class['name'], top_class['name'])} ({confidence*100:.0f}%), abaixo do limiar mínimo",
            "Caso tenha sintomas, procure avaliação profissional",
        ]

        return {
            "classification": "Normal",
            "confidence": round(normal_score, 4),
            "scores": scores,
            "findings": findings,
            "systemic_alert": None,
            "urgency": "none",
            "recommendation": "Nenhuma condição identificada com certeza. Se houver sintomas, consulte um oftalmologista.",
            "model": "MobileNetV2-TFLite",
            "description": f"Nenhuma condição detectada com confiança acima de {int(CONFIDENCE_THRESHOLD*100)}%. Aparenta normal."
        }

    # Clinical findings based on classification
    findings = generate_findings(top_class["name"], confidence, scores)

    return {
        "classification": top_class["name"],
        "confidence": round(confidence, 4),
        "scores": scores,
        "findings": findings,
        "systemic_alert": top_class["systemic"] if confidence > 0.5 else None,
        "urgency": top_class["urgency"],
        "recommendation": generate_recommendation(top_class["name"], confidence),
        "model": "MobileNetV2-TFLite",
        "description": f"Inferência ML: {CLASS_PT.get(top_class['name'], top_class['name'])} ({confidence*100:.1f}% de confiança)"
    }


def generate_findings(cls: str, conf: float, scores: dict) -> list:
    findings_map = {
        "Normal":         ["Esclera aparenta estar limpa", "Sem anormalidades visíveis detectadas", "Pupilas aparentam simétricas"],
        "Cataract":       ["Opacidade ou turvação do cristalino detectada", "Área pupilar com claridade reduzida", "Possível indicador de comprometimento visual"],
        "Conjunctivitis": ["Vermelhidão conjuntival detectada", "Possível inflamação da conjuntiva", "Monitorar secreção ou lacrimejamento"],
        "Hemorrhage":     ["Hemorragia subconjuntival detectada", "Sangue visível na área escleral", "Pode indicar problema vascular"],
        "Jaundice":       ["Amarelamento da esclera detectado", "Possível elevação de bilirrubina", "Avaliação sistêmica recomendada"],
        "Pterygium":      ["Crescimento de tecido na conjuntiva detectado", "Crescimento se estendendo em direção à córnea", "Proteção UV recomendada"],
        "Ptosis":         ["Queda da pálpebra detectada", "Posicionamento assimétrico das pálpebras", "Avaliação neurológica pode ser necessária"],
        "Stye/Chalazion": ["Nódulo ou inchaço na pálpebra detectado", "Possível inflamação glandular", "Compressa morna recomendada"],
        "Uveitis":        ["Vermelhidão ocular profunda detectada", "Possível inflamação da úvea", "Avaliação oftalmológica urgente necessária"],
    }
    base = findings_map.get(cls, ["Análise concluída"])
    if conf < 0.5:
        base.append(f"Confiança baixa ({conf*100:.0f}%) — recomenda-se avaliação profissional")
    return base


# Portuguese class name mapping for descriptions
CLASS_PT = {
    "Normal": "Normal", "Cataract": "Catarata", "Conjunctivitis": "Conjuntivite",
    "Hemorrhage": "Hemorragia", "Jaundice": "Icterícia", "Pterygium": "Pterígio",
    "Ptosis": "Ptose", "Stye/Chalazion": "Terçol/Calázio", "Uveitis": "Uveíte",
}


def generate_recommendation(cls: str, conf: float) -> str:
    pt = CLASS_PT.get(cls, cls)
    if cls == "Normal" and conf > 0.7:
        return "Sem preocupações imediatas. Continue com exames oftalmológicos regulares."
    elif cls == "Normal":
        return "Aparenta normal, mas com baixa confiança. Considere avaliação profissional se houver sintomas."
    elif conf > 0.6:
        return f"{pt} detectado(a) com boa confiança. Consulte um oftalmologista para avaliação adequada."
    else:
        return f"Possível {pt} detectado(a). Avaliação profissional recomendada para diagnóstico definitivo."


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__) or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return JSONResponse({"service": "EYECHECK V2.1 API", "model": MODEL_PATH, "classes": 9, "status": "running"})


@app.get("/api")
async def api_info():
    return {"service": "EYECHECK V2.1 API", "model": MODEL_PATH, "classes": 9, "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze eye photo and return 9-class classification"""
    try:
        image_bytes = await file.read()
        result = run_inference(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Failed to process image"}
        )


@app.post("/analyze-base64")
async def analyze_base64(data: dict):
    """Analyze base64-encoded eye photo"""
    import base64
    try:
        b64 = data.get("image", "")
        # Remove data URL prefix if present
        if "," in b64:
            b64 = b64.split(",")[1]
        image_bytes = base64.b64decode(b64)
        result = run_inference(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Failed to process image"}
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
