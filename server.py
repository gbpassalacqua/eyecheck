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
        normal_info = CLASS_INFO.get("Normal", {})

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
            "explanation": normal_info.get("explanation", ""),
            "severity": normal_info.get("severity", ""),
            "severity_level": normal_info.get("severity_level", "none"),
            "systemic_alert": None,
            "urgency": "none",
            "recommendation": "Nenhuma condição identificada com certeza. Se houver sintomas, consulte um oftalmologista.",
            "model": "MobileNetV2-TFLite",
            "description": f"Nenhuma condição detectada com confiança acima de {int(CONFIDENCE_THRESHOLD*100)}%. Aparenta normal."
        }

    # Clinical findings based on classification
    findings = generate_findings(top_class["name"], confidence, scores)
    info = CLASS_INFO.get(top_class["name"], {})

    return {
        "classification": top_class["name"],
        "confidence": round(confidence, 4),
        "scores": scores,
        "findings": findings,
        "explanation": info.get("explanation", ""),
        "severity": info.get("severity", ""),
        "severity_level": info.get("severity_level", "none"),
        "systemic_alert": top_class["systemic"] if confidence > 0.5 else None,
        "urgency": top_class["urgency"],
        "recommendation": generate_recommendation(top_class["name"], confidence),
        "model": "MobileNetV2-TFLite",
        "description": f"Inferência ML: {CLASS_PT.get(top_class['name'], top_class['name'])} ({confidence*100:.1f}% de confiança)"
    }


def generate_findings(cls: str, conf: float, scores: dict) -> list:
    """Generate findings — only what the AI detected, NO recommendations here"""
    findings_map = {
        "Normal":         ["Parte branca do olho aparenta limpa", "Nenhuma alteração visível detectada", "Coloração e formato dentro do esperado"],
        "Cataract":       ["Possível opacidade na região do cristalino", "Área da pupila com claridade reduzida", "Pode afetar a nitidez da visão"],
        "Conjunctivitis": ["Vermelhidão na superfície do olho detectada", "Possível inflamação da conjuntiva", "Pode haver secreção ou lacrimejamento"],
        "Hemorrhage":     ["Mancha de sangue na parte branca do olho", "Vasos sanguíneos aparentam rompidos", "Aparência semelhante a um roxo na pele"],
        "Jaundice":       ["Amarelamento na parte branca do olho", "Coloração diferente do esperado", "Pode estar relacionado ao fígado"],
        "Pterygium":      ["Crescimento de tecido na superfície do olho", "Tecido se estendendo em direção à córnea", "Relacionado à exposição solar"],
        "Ptosis":         ["Pálpebra superior mais caída que o normal", "Assimetria entre as pálpebras", "Pode ser por cansaço ou outras causas"],
        "Stye/Chalazion": ["Nódulo ou inchaço na pálpebra detectado", "Possível glândula inflamada", "Comum e geralmente temporário"],
        "Uveitis":        ["Vermelhidão profunda no olho detectada", "Possível inflamação interna", "Diferente de conjuntivite comum"],
    }
    base = findings_map.get(cls, ["Análise concluída"])
    if conf < 0.5:
        base.append(f"Confiança da IA: {conf*100:.0f}% — resultado incerto")
    return base


# Portuguese class name mapping for descriptions
CLASS_PT = {
    "Normal": "Normal", "Cataract": "Catarata", "Conjunctivitis": "Conjuntivite",
    "Hemorrhage": "Hemorragia", "Jaundice": "Icterícia", "Pterygium": "Pterígio",
    "Ptosis": "Ptose", "Stye/Chalazion": "Terçol/Calázio", "Uveitis": "Uveíte",
}

# Explanation of each condition in plain language + severity level
# NOTE: Explanations must NOT include recommendations (there's a separate field for that)
CLASS_INFO = {
    "Normal": {
        "explanation": "Seus olhos aparentam estar saudáveis. Nenhuma alteração visível foi detectada pela IA.",
        "severity": "Nenhuma preocupação",
        "severity_level": "none",
    },
    "Cataract": {
        "explanation": "Catarata é quando o cristalino (a lente natural do olho) fica opaco. É muito comum com a idade e tem tratamento simples. Não é uma emergência.",
        "severity": "Não é grave — tem tratamento",
        "severity_level": "moderate",
    },
    "Conjunctivitis": {
        "explanation": "Conjuntivite é uma inflamação que deixa o olho vermelho. Pode ser causada por vírus, bactéria ou alergia. Na maioria dos casos melhora sozinha ou com colírios simples.",
        "severity": "Não é grave — geralmente resolve sozinha",
        "severity_level": "low",
    },
    "Hemorrhage": {
        "explanation": "É um pequeno derramamento de sangue na parte branca do olho. Parece assustador, mas é inofensivo e desaparece sozinho em 1 a 2 semanas. É como um roxo na pele.",
        "severity": "Não é grave — desaparece sozinho",
        "severity_level": "low",
    },
    "Jaundice": {
        "explanation": "Icterícia é o amarelamento da parte branca do olho. Pode estar relacionado ao fígado. Vale a pena fazer exames para investigar.",
        "severity": "Atenção — vale investigar com exames",
        "severity_level": "high",
    },
    "Pterygium": {
        "explanation": "Pterígio é um crescimento benigno (não é câncer) na superfície do olho. É comum em quem pega muito sol. Na maioria dos casos basta usar óculos de sol para proteger.",
        "severity": "Não é grave — condição benigna",
        "severity_level": "low",
    },
    "Ptosis": {
        "explanation": "Ptose é quando a pálpebra superior fica mais caída que o normal. Pode ser por cansaço, idade, ou em casos raros algo neurológico. Se for persistente, vale avaliar.",
        "severity": "Geralmente não é grave — observar se persistir",
        "severity_level": "moderate",
    },
    "Stye/Chalazion": {
        "explanation": "Terçol ou calázio é um carocinho na pálpebra causado por uma glândula inflamada. É muito comum e não é grave. Compressas mornas ajudam a resolver.",
        "severity": "Não é grave — muito comum e tratável",
        "severity_level": "low",
    },
    "Uveitis": {
        "explanation": "Uveíte é uma inflamação dentro do olho que pode causar vermelhidão e sensibilidade à luz. É tratável com colírios anti-inflamatórios e tem bom prognóstico.",
        "severity": "Tratável — colírios resolvem na maioria dos casos",
        "severity_level": "moderate",
    },
}


def generate_recommendation(cls: str, conf: float) -> str:
    """Generate specific recommendation per condition — human-friendly language"""
    rec_map = {
        "Normal":         "Tudo certo! Mantenha seus exames de rotina em dia.",
        "Cataract":       "Catarata é comum e tem solução. Quando puder, agende uma consulta oftalmológica para acompanhar.",
        "Conjunctivitis": "Evite coçar o olho e lave as mãos com frequência. Se a vermelhidão persistir por mais de 3 dias, procure um médico.",
        "Hemorrhage":     "Fique tranquilo, geralmente desaparece sozinho. Se acontecer com frequência, vale medir a pressão arterial.",
        "Jaundice":       "Vale a pena fazer exames de sangue para checar o fígado. Procure um clínico geral.",
        "Pterygium":      "Use óculos de sol com proteção UV. Se crescer ou incomodar a visão, procure um oftalmologista.",
        "Ptosis":         "Se a pálpebra caída for recente ou piorar, procure avaliação médica. Se for antigo, geralmente é benigno.",
        "Stye/Chalazion": "Aplique compressas mornas por 10 minutos, 3x ao dia. Se não melhorar em 2 semanas, procure um oftalmologista.",
        "Uveitis":        "Procure um oftalmologista para avaliação. O tratamento com colírios costuma resolver bem.",
    }
    base = rec_map.get(cls, "Consulte um profissional de saúde para avaliação.")
    if conf < 0.5 and cls != "Normal":
        base = "A IA não tem certeza desse resultado. " + base
    return base


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
