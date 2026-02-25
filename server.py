"""
EYECHECK V3.0 — Dual Model TFLite Inference API
V2: Ocular diseases (9 classes)
V3: Systemic diseases visible through the eyes (9 classes)
Tagline: "Seu olho conta a saúde do seu corpo"
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
from PIL import Image
import io
import os
import uuid
from datetime import datetime, timezone

# TFLite runtime — triple fallback
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

app = FastAPI(title="EYECHECK V3.0 API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# CLASS DEFINITIONS (alphabetical = model output index order)
# ═══════════════════════════════════════════════════════════════

V2_CLASSES = [
    "cataract", "conjunctivitis", "hemorrhage", "jaundice", "normal",
    "pterygium", "ptosis", "stye_chalazion", "uveitis",
]

V3_CLASSES = [
    "anemia", "arcus_senilis", "blue_sclera", "diabetic_eye", "exophthalmos",
    "kayser_fleischer", "normal", "periorbital_edema", "xanthelasma",
]

# ═══════════════════════════════════════════════════════════════
# URGENCY
# ═══════════════════════════════════════════════════════════════

URGENCY_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3}

V2_URGENCY = {
    "normal": "none", "pterygium": "low", "stye_chalazion": "low",
    "cataract": "moderate", "conjunctivitis": "moderate", "ptosis": "moderate",
    "hemorrhage": "high", "jaundice": "high", "uveitis": "high",
}

V3_URGENCY = {
    "normal": "none",
    "arcus_senilis": "moderate", "diabetic_eye": "moderate", "xanthelasma": "moderate",
    "anemia": "high", "blue_sclera": "high", "exophthalmos": "high",
    "kayser_fleischer": "high", "periorbital_edema": "high",
}

# ═══════════════════════════════════════════════════════════════
# PORTUGUESE NAMES
# ═══════════════════════════════════════════════════════════════

V2_PT = {
    "normal": "Normal", "cataract": "Catarata", "conjunctivitis": "Conjuntivite",
    "hemorrhage": "Hemorragia", "jaundice": "Icterícia", "pterygium": "Pterígio",
    "ptosis": "Ptose", "stye_chalazion": "Terçol/Calázio", "uveitis": "Uveíte",
}

V3_PT = {
    "normal": "Normal", "anemia": "Anemia", "arcus_senilis": "Arco Senil",
    "blue_sclera": "Esclera Azulada", "diabetic_eye": "Olho Diabético",
    "exophthalmos": "Exoftalmia", "kayser_fleischer": "Anel de Kayser-Fleischer",
    "periorbital_edema": "Edema Periorbital", "xanthelasma": "Xantelasma",
}

# ═══════════════════════════════════════════════════════════════
# V2 OCULAR — EXPLANATIONS, FINDINGS, RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

V2_INFO = {
    "normal": {
        "explanation": "Seus olhos aparentam estar saudáveis. Nenhuma alteração ocular visível foi detectada.",
        "severity": "Nenhuma preocupação",
        "severity_level": "none",
    },
    "cataract": {
        "explanation": "Catarata é quando o cristalino (a lente natural do olho) fica opaco. É muito comum com a idade e tem tratamento simples. Não é uma emergência.",
        "severity": "Não é grave — tem tratamento",
        "severity_level": "moderate",
    },
    "conjunctivitis": {
        "explanation": "Conjuntivite é uma inflamação que deixa o olho vermelho. Pode ser por vírus, bactéria ou alergia. Na maioria dos casos melhora sozinha ou com colírios simples.",
        "severity": "Não é grave — geralmente resolve sozinha",
        "severity_level": "low",
    },
    "hemorrhage": {
        "explanation": "É um pequeno derramamento de sangue na parte branca do olho. Parece assustador, mas é inofensivo e desaparece sozinho em 1 a 2 semanas. É como um roxo na pele.",
        "severity": "Não é grave — desaparece sozinho",
        "severity_level": "low",
    },
    "jaundice": {
        "explanation": "Icterícia é o amarelamento da parte branca do olho. Pode estar relacionado ao fígado. Vale a pena fazer exames para investigar.",
        "severity": "Atenção — vale investigar com exames",
        "severity_level": "high",
    },
    "pterygium": {
        "explanation": "Pterígio é um crescimento benigno (não é câncer) na superfície do olho. É comum em quem pega muito sol. Na maioria dos casos basta usar óculos de sol.",
        "severity": "Não é grave — condição benigna",
        "severity_level": "low",
    },
    "ptosis": {
        "explanation": "Ptose é quando a pálpebra superior fica mais caída que o normal. Pode ser por cansaço, idade, ou em casos raros algo neurológico. Se for persistente, vale avaliar.",
        "severity": "Geralmente não é grave — observar se persistir",
        "severity_level": "moderate",
    },
    "stye_chalazion": {
        "explanation": "Terçol ou calázio é um carocinho na pálpebra causado por uma glândula inflamada. É muito comum e não é grave. Compressas mornas ajudam a resolver.",
        "severity": "Não é grave — muito comum e tratável",
        "severity_level": "low",
    },
    "uveitis": {
        "explanation": "Uveíte é uma inflamação dentro do olho que pode causar vermelhidão e sensibilidade à luz. É tratável com colírios anti-inflamatórios e tem bom prognóstico.",
        "severity": "Tratável — colírios resolvem na maioria dos casos",
        "severity_level": "moderate",
    },
}

V2_FINDINGS = {
    "normal":         ["Parte branca do olho aparenta limpa", "Nenhuma alteração visível detectada", "Coloração e formato dentro do esperado"],
    "cataract":       ["Possível opacidade na região do cristalino", "Área da pupila com claridade reduzida", "Pode afetar a nitidez da visão"],
    "conjunctivitis": ["Vermelhidão na superfície do olho detectada", "Possível inflamação da conjuntiva", "Pode haver secreção ou lacrimejamento"],
    "hemorrhage":     ["Mancha de sangue na parte branca do olho", "Vasos sanguíneos aparentam rompidos", "Aparência semelhante a um roxo na pele"],
    "jaundice":       ["Amarelamento na parte branca do olho", "Coloração diferente do esperado", "Pode estar relacionado ao fígado"],
    "pterygium":      ["Crescimento de tecido na superfície do olho", "Tecido se estendendo em direção à córnea", "Relacionado à exposição solar"],
    "ptosis":         ["Pálpebra superior mais caída que o normal", "Assimetria entre as pálpebras", "Pode ser por cansaço ou outras causas"],
    "stye_chalazion": ["Nódulo ou inchaço na pálpebra detectado", "Possível glândula inflamada", "Comum e geralmente temporário"],
    "uveitis":        ["Vermelhidão profunda no olho detectada", "Possível inflamação interna", "Diferente de conjuntivite comum"],
}

V2_RECOMMENDATIONS = {
    "normal":         "Tudo certo! Mantenha seus exames de rotina em dia.",
    "cataract":       "Catarata é comum e tem solução. Quando puder, agende uma consulta oftalmológica.",
    "conjunctivitis": "Evite coçar o olho e lave as mãos com frequência. Se persistir por mais de 3 dias, procure um médico.",
    "hemorrhage":     "Fique tranquilo, geralmente desaparece sozinho. Se acontecer com frequência, meça a pressão arterial.",
    "jaundice":       "Vale a pena fazer exames de sangue para checar o fígado. Procure um clínico geral.",
    "pterygium":      "Use óculos de sol com proteção UV. Se crescer ou incomodar a visão, procure um oftalmologista.",
    "ptosis":         "Se a pálpebra caída for recente ou piorar, procure avaliação médica. Se for antigo, geralmente é benigno.",
    "stye_chalazion": "Aplique compressas mornas por 10 minutos, 3x ao dia. Se não melhorar em 2 semanas, procure um oftalmologista.",
    "uveitis":        "Procure um oftalmologista para avaliação. O tratamento com colírios costuma resolver bem.",
}

# ═══════════════════════════════════════════════════════════════
# V3 SYSTEMIC — EXPLANATIONS, FINDINGS, RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

V3_INFO = {
    "normal": {
        "explanation": "Nenhum sinal de doença sistêmica detectado através dos olhos. Aparenta tudo normal.",
        "severity": "Nenhuma preocupação",
        "severity_level": "none",
        "disease": None,
        "sign": "Olho saudável",
    },
    "anemia": {
        "explanation": "Anemia é quando o sangue tem poucos glóbulos vermelhos. Pode causar cansaço e palidez. É muito comum e tratável com alimentação e suplementos de ferro.",
        "severity": "Tratável — exame de sangue confirma",
        "severity_level": "high",
        "disease": "Anemia ferropriva, deficiência de B12",
        "sign": "Conjuntiva pálida (parte interna da pálpebra esbranquiçada)",
    },
    "arcus_senilis": {
        "explanation": "Arco senil é um anel branco ao redor da íris. Em pessoas acima de 60 anos é normal. Em mais jovens, pode indicar colesterol alto. Vale checar com exames.",
        "severity": "Geralmente benigno — checar colesterol se jovem",
        "severity_level": "moderate",
        "disease": "Colesterol alto, hiperlipidemia",
        "sign": "Anel branco/acinzentado ao redor da íris",
    },
    "blue_sclera": {
        "explanation": "Esclera azulada é quando a parte branca do olho tem um tom azulado. Pode ser uma variação normal ou indicar condições genéticas. Vale avaliação médica.",
        "severity": "Vale avaliar — pode ser variação normal",
        "severity_level": "high",
        "disease": "Osteogênese imperfeita, Ehlers-Danlos, anemia severa",
        "sign": "Parte branca do olho com tom azulado",
    },
    "diabetic_eye": {
        "explanation": "Sinais oculares que podem indicar diabetes. O diabetes afeta os vasos do olho ao longo do tempo. Controle da glicemia previne complicações.",
        "severity": "Atenção — controlar glicemia é essencial",
        "severity_level": "moderate",
        "disease": "Diabetes mellitus tipo 1 ou 2",
        "sign": "Alterações vasculares na superfície ocular",
    },
    "exophthalmos": {
        "explanation": "Exoftalmia é quando os olhos ficam mais saltados que o normal. A causa mais comum é o hipertireoidismo (doença de Graves). É tratável com acompanhamento médico.",
        "severity": "Tratável — procurar endocrinologista",
        "severity_level": "high",
        "disease": "Hipertireoidismo (Doença de Graves)",
        "sign": "Olhos proeminentes/saltados",
    },
    "kayser_fleischer": {
        "explanation": "Anel de Kayser-Fleischer é um anel dourado/marrom ao redor da córnea. É raro e pode indicar doença de Wilson (acúmulo de cobre no corpo).",
        "severity": "Raro — avaliação médica recomendada",
        "severity_level": "high",
        "disease": "Doença de Wilson (acúmulo de cobre)",
        "sign": "Anel dourado/marrom ao redor da córnea",
    },
    "periorbital_edema": {
        "explanation": "Edema periorbital é um inchaço ao redor dos olhos. Pode ter causas simples (alergia, sono ruim) ou indicar problemas renais. Se for persistente, vale investigar.",
        "severity": "Pode ser simples — investigar se persistente",
        "severity_level": "high",
        "disease": "Doença renal, hipotireoidismo, nefrose",
        "sign": "Inchaço/bolsas ao redor dos olhos",
    },
    "xanthelasma": {
        "explanation": "Xantelasma são placas amareladas na pálpebra. São depósitos de gordura e podem indicar colesterol alto. São benignos mas vale checar os lipídeos.",
        "severity": "Benigno — checar colesterol",
        "severity_level": "moderate",
        "disease": "Colesterol alto, dislipidemia, risco cardíaco",
        "sign": "Placas amareladas na pálpebra",
    },
}

V3_FINDINGS = {
    "normal":            ["Nenhum sinal sistêmico detectado", "Coloração da conjuntiva normal", "Sem alterações perioculares visíveis"],
    "anemia":            ["Palidez na conjuntiva detectada", "Parte interna da pálpebra esbranquiçada", "Possível baixo nível de hemoglobina"],
    "arcus_senilis":     ["Anel branco/acinzentado ao redor da íris", "Depósito lipídico na periferia da córnea", "Pode indicar colesterol elevado em jovens"],
    "blue_sclera":       ["Tom azulado na parte branca do olho", "Esclera com coloração incomum", "Possível afinamento da esclera"],
    "diabetic_eye":      ["Alterações vasculares na superfície ocular", "Possíveis sinais externos de diabetes", "Microangiopatia pode estar presente"],
    "exophthalmos":      ["Proeminência ocular detectada", "Olhos aparentam mais saltados", "Possível relação com tireoide"],
    "kayser_fleischer":  ["Anel dourado/marrom na periferia da córnea", "Depósito de cobre visível", "Sinal raro mas clinicamente significativo"],
    "periorbital_edema": ["Inchaço ao redor dos olhos detectado", "Edema periorbital visível", "Pode ter diversas causas"],
    "xanthelasma":       ["Placas amareladas na região da pálpebra", "Depósitos de gordura subcutâneos", "Possível indicador de colesterol alto"],
}

V3_RECOMMENDATIONS = {
    "normal":            "Nenhum sinal de doença sistêmica. Continue com seus exames de rotina.",
    "anemia":            "Procure um clínico geral e peça um hemograma completo. Alimentação rica em ferro e B12 ajuda.",
    "arcus_senilis":     "Se você tem menos de 50 anos, vale pedir exame de colesterol. Acima de 60 anos é geralmente normal.",
    "blue_sclera":       "Procure avaliação médica para investigar a causa. Pode ser variação normal ou algo genético.",
    "diabetic_eye":      "Verifique sua glicemia com um clínico geral. Controlar o açúcar no sangue previne complicações.",
    "exophthalmos":      "Procure um endocrinologista para checar a tireoide. Doença de Graves é tratável.",
    "kayser_fleischer":  "Procure um hepatologista ou clínico geral. Exames de cobre e ceruloplasmina podem confirmar.",
    "periorbital_edema": "Se o inchaço for persistente, procure um nefrologista. Pode ser alergia, sono ruim, ou algo renal.",
    "xanthelasma":       "Peça exame de colesterol e triglicerídeos. Mudanças na dieta e exercícios podem ajudar.",
}

# ═══════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLD
# ═══════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.50

# ═══════════════════════════════════════════════════════════════
# LOAD BOTH MODELS AT STARTUP
# ═══════════════════════════════════════════════════════════════

V2_MODEL_PATH = os.environ.get("V2_MODEL_PATH", "guard2live_v2_9class.tflite")
V3_MODEL_PATH = os.environ.get("V3_MODEL_PATH", "guard2live_v3_systemic.tflite")

v2_interpreter = tflite.Interpreter(model_path=V2_MODEL_PATH)
v2_interpreter.allocate_tensors()
v2_input = v2_interpreter.get_input_details()
v2_output = v2_interpreter.get_output_details()

v3_interpreter = tflite.Interpreter(model_path=V3_MODEL_PATH)
v3_interpreter.allocate_tensors()
v3_input = v3_interpreter.get_input_details()
v3_output = v3_interpreter.get_output_details()

INPUT_SIZE = v2_input[0]['shape'][1]  # 224

print(f"V2 Ocular loaded: {V2_MODEL_PATH} | {len(V2_CLASSES)} classes | input {INPUT_SIZE}x{INPUT_SIZE}")
print(f"V3 Systemic loaded: {V3_MODEL_PATH} | {len(V3_CLASSES)} classes | input {INPUT_SIZE}x{INPUT_SIZE}")


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING & INFERENCE
# ═══════════════════════════════════════════════════════════════

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # MobileNetV2 range [-1, 1]
    return np.expand_dims(arr, axis=0)


def run_model(interpreter, inp_details, out_details, tensor, class_names):
    interpreter.set_tensor(inp_details[0]['index'], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(out_details[0]['index'])[0]
    scores = {name: round(float(s), 4) for name, s in zip(class_names, output)}
    top_idx = int(np.argmax(output))
    return class_names[top_idx], float(output[top_idx]), scores


def run_ocular(image_bytes: bytes) -> dict:
    tensor = preprocess(image_bytes)
    top_class, confidence, scores = run_model(v2_interpreter, v2_input, v2_output, tensor, V2_CLASSES)

    # Threshold: below 50% → report as normal
    if confidence < CONFIDENCE_THRESHOLD and top_class != "normal":
        original = top_class
        info = V2_INFO.get("normal", {})
        return {
            "model": "Guard2Live V2 — Ocular",
            "type": "ocular",
            "top_prediction": "normal",
            "top_prediction_pt": "Normal",
            "confidence": round(scores.get("normal", 0), 4),
            "all_scores": scores,
            "urgency": "none",
            "findings": [
                "Nenhuma condição ocular detectada com confiança suficiente",
                "Aparência geral dentro da normalidade",
                f"Maior suspeita: {V2_PT.get(original, original)} ({confidence*100:.0f}%), abaixo do limiar",
            ],
            "explanation": info.get("explanation", ""),
            "severity": info.get("severity", ""),
            "severity_level": info.get("severity_level", "none"),
            "recommendation": "Nenhuma condição ocular identificada com certeza. Se houver sintomas, consulte um oftalmologista.",
        }

    info = V2_INFO.get(top_class, {})
    return {
        "model": "Guard2Live V2 — Ocular",
        "type": "ocular",
        "top_prediction": top_class,
        "top_prediction_pt": V2_PT.get(top_class, top_class),
        "confidence": round(confidence, 4),
        "all_scores": scores,
        "urgency": V2_URGENCY.get(top_class, "none"),
        "findings": list(V2_FINDINGS.get(top_class, ["Análise concluída"])),
        "explanation": info.get("explanation", ""),
        "severity": info.get("severity", ""),
        "severity_level": info.get("severity_level", "none"),
        "recommendation": V2_RECOMMENDATIONS.get(top_class, "Consulte um profissional."),
    }


def run_systemic(image_bytes: bytes) -> dict:
    tensor = preprocess(image_bytes)
    top_class, confidence, scores = run_model(v3_interpreter, v3_input, v3_output, tensor, V3_CLASSES)

    # Threshold: below 50% → report as normal
    if confidence < CONFIDENCE_THRESHOLD and top_class != "normal":
        original = top_class
        info = V3_INFO.get("normal", {})
        return {
            "model": "Guard2Live V3 — Sistêmico",
            "type": "systemic",
            "top_prediction": "normal",
            "top_prediction_pt": "Normal",
            "confidence": round(scores.get("normal", 0), 4),
            "all_scores": scores,
            "urgency": "none",
            "findings": [
                "Nenhum sinal sistêmico detectado com confiança suficiente",
                "Aparência geral dentro da normalidade",
                f"Maior suspeita: {V3_PT.get(original, original)} ({confidence*100:.0f}%), abaixo do limiar",
            ],
            "explanation": info.get("explanation", ""),
            "severity": info.get("severity", ""),
            "severity_level": info.get("severity_level", "none"),
            "systemic_alert": None,
            "recommendation": "Nenhum sinal sistêmico identificado com certeza.",
        }

    info = V3_INFO.get(top_class, {})
    systemic_alert = None
    if top_class != "normal":
        systemic_alert = {
            "disease": info.get("disease", ""),
            "sign": info.get("sign", ""),
        }

    return {
        "model": "Guard2Live V3 — Sistêmico",
        "type": "systemic",
        "top_prediction": top_class,
        "top_prediction_pt": V3_PT.get(top_class, top_class),
        "confidence": round(confidence, 4),
        "all_scores": scores,
        "urgency": V3_URGENCY.get(top_class, "none"),
        "findings": list(V3_FINDINGS.get(top_class, ["Análise concluída"])),
        "explanation": info.get("explanation", ""),
        "severity": info.get("severity", ""),
        "severity_level": info.get("severity_level", "none"),
        "systemic_alert": systemic_alert,
        "recommendation": V3_RECOMMENDATIONS.get(top_class, "Consulte um profissional."),
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__) or ".", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return JSONResponse({"service": "EYECHECK V3.0", "models": 2, "status": "running"})


@app.get("/health")
async def health():
    return {"status": "ok", "models": {"v2": V2_MODEL_PATH, "v3": V3_MODEL_PATH}}


@app.get("/api")
async def api_info():
    return {
        "service": "EYECHECK V3.0 API",
        "models": {"v2_ocular": V2_MODEL_PATH, "v3_systemic": V3_MODEL_PATH},
        "v2_classes": V2_CLASSES,
        "v3_classes": V3_CLASSES,
        "status": "running",
    }


def _decode_image(data: dict) -> bytes:
    import base64
    b64 = data.get("image", "")
    if "," in b64:
        b64 = b64.split(",")[1]
    return base64.b64decode(b64)


@app.post("/predict/full")
async def predict_full(data: dict):
    """Run BOTH models on the same image — primary endpoint"""
    try:
        image_bytes = _decode_image(data)
        ocular = run_ocular(image_bytes)
        systemic = run_systemic(image_bytes)

        # Combined urgency = max of both
        v2_urg = ocular["urgency"]
        v3_urg = systemic["urgency"]
        combined = max(v2_urg, v3_urg, key=lambda x: URGENCY_ORDER.get(x, 0))

        return JSONResponse(content={
            "image_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ocular_analysis": ocular,
            "systemic_analysis": systemic,
            "combined_urgency": combined,
            "disclaimer": "Triagem de bem-estar. Não substitui diagnóstico médico profissional.",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/ocular")
async def predict_ocular(data: dict):
    """Run V2 Ocular model only"""
    try:
        image_bytes = _decode_image(data)
        return JSONResponse(content=run_ocular(image_bytes))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/systemic")
async def predict_systemic(data: dict):
    """Run V3 Systemic model only"""
    try:
        image_bytes = _decode_image(data)
        return JSONResponse(content=run_systemic(image_bytes))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Backward compatibility
@app.post("/analyze-base64")
async def analyze_base64(data: dict):
    return await predict_full(data)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
