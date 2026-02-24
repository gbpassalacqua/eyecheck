"""
Guard2Live V2.1 — TFLite Inference API
Deploy on Render, Railway, or any Python host
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

app = FastAPI(title="Guard2Live V2.1 API", version="2.1.0")

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

    # Clinical findings based on classification
    findings = generate_findings(top_class["name"], confidence, scores)

    return {
        "classification": top_class["name"],
        "confidence": round(confidence, 4),
        "scores": scores,
        "findings": findings,
        "systemic_alert": top_class["systemic"] if confidence > 0.3 else None,
        "urgency": top_class["urgency"],
        "recommendation": generate_recommendation(top_class["name"], confidence),
        "model": "MobileNetV2-TFLite",
        "description": f"TFLite inference: {top_class['name']} ({confidence*100:.1f}% confidence)"
    }


def generate_findings(cls: str, conf: float, scores: dict) -> list:
    findings_map = {
        "Normal":         ["Sclera appears clear", "No visible abnormalities detected", "Pupils appear symmetric"],
        "Cataract":       ["Lens opacity or cloudiness detected", "Pupil area shows reduced clarity", "Possible visual impairment indicator"],
        "Conjunctivitis": ["Conjunctival redness detected", "Possible inflammation of conjunctiva", "Monitor for discharge or tearing"],
        "Hemorrhage":     ["Subconjunctival hemorrhage detected", "Blood visible in scleral area", "May indicate vascular issue"],
        "Jaundice":       ["Scleral yellowing detected", "Possible elevated bilirubin", "Systemic evaluation recommended"],
        "Pterygium":      ["Tissue growth on conjunctiva detected", "Growth extending toward cornea", "UV protection recommended"],
        "Ptosis":         ["Eyelid drooping detected", "Asymmetric lid positioning", "Neurological evaluation may be needed"],
        "Stye/Chalazion": ["Eyelid nodule or swelling detected", "Possible glandular inflammation", "Warm compress recommended"],
        "Uveitis":        ["Deep ocular redness detected", "Possible uveal inflammation", "Prompt ophthalmologic evaluation needed"],
    }
    base = findings_map.get(cls, ["Analysis complete"])
    if conf < 0.5:
        base.append(f"Low confidence ({conf*100:.0f}%) — recommend professional evaluation")
    return base


def generate_recommendation(cls: str, conf: float) -> str:
    if cls == "Normal" and conf > 0.7:
        return "No immediate concerns. Continue regular eye check-ups."
    elif cls == "Normal":
        return "Appears normal but low confidence. Consider professional evaluation if symptoms present."
    elif conf > 0.6:
        return f"{cls} detected with good confidence. Consult an ophthalmologist for proper evaluation."
    else:
        return f"Possible {cls} detected. Professional evaluation recommended for definitive diagnosis."


@app.get("/")
async def root():
    return {"service": "Guard2Live V2.1 API", "model": MODEL_PATH, "classes": 9, "status": "running"}


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
